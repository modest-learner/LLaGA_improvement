import sys
sys.path.append("./")
sys.path.append("./utils")
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from utils.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN
from utils.conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model
from utils.utils import disable_torch_init, tokenizer_graph_token, get_model_name_from_path
from torch_geometric.utils import k_hop_subgraph, degree, remove_self_loops, add_self_loops
from torch_geometric.nn import MessagePassing
import math
import hashlib
from transformers import set_seed
from eval.lp_inputs import donors_for_pair, pair_ho, intervene, endpoint_prompt

SMALL_DATASETS=["pubmed", "cora"]


class MP(MessagePassing):
    # PyG cannot infer these arguments from propagate() calls outside the class.
    propagate_type = {'x': torch.Tensor, 'norm': torch.Tensor}

    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


# def get_chunk(lst, n, k):
#     chunks = split_list(lst, n)
#     return chunks[k]

def load_pretrain_embedding_graph(data_dir, pretrained_embedding_type):
    if pretrained_embedding_type == "simteg":
        simteg_sbert = torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))
        simteg_roberta = torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))
        simteg_e5 = torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))
        pretrained_emb = torch.concat([simteg_sbert, simteg_roberta, simteg_e5], dim=-1)
    else:
        pretrained_emb = torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))
    return pretrained_emb

def load_pretrain_embedding_hop(data_dir, pretrained_embedding_type, hop, mask):
    if pretrained_embedding_type == "simteg":
        simteg_sbert=[torch.load(os.path.join(data_dir, f"simteg_sbert_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)]
        simteg_roberta = [torch.load(os.path.join(data_dir, f"simteg_roberta_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_roberta_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)]
        simteg_e5 = [torch.load(os.path.join(data_dir, f"simteg_e5_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)]
        pretrained_embs = [torch.cat([simteg_sbert[i], simteg_roberta[i], simteg_e5[i]], dim=-1) for i in range(hop + 1)]
    else:
        pretrained_embs = [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))[mask]]+  [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_{i}hop_x.pt"))[mask] for i in range(1, hop+1)]

    return pretrained_embs

def load_pretrain_embedding_hop_lp(data_dir, pretrained_embedding_type, hop):
    mask = torch.load(os.path.join(data_dir, f"no_test_link_mask.pt"))
    if pretrained_embedding_type == "simteg":
        simteg_sbert=[torch.load(os.path.join(data_dir, f"simteg_sbert_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x_notestlink.pt")) for i in range(1, hop + 1)]
        simteg_roberta = [torch.load(os.path.join(data_dir, f"simteg_roberta_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_roberta_{i}hop_x_notestlink.pt")) for i in range(1, hop + 1)]
        simteg_e5 = [torch.load(os.path.join(data_dir, f"simteg_e5_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x_notestlink.pt")) for i in range(1, hop + 1)]
        pretrained_embs = [torch.cat([simteg_sbert[i], simteg_roberta[i], simteg_e5[i]], dim=-1) for i in range(hop + 1)]
    else:
        pretrained_embs = [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))[mask]]+  [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_{i}hop_x_notestlink.pt")) for i in range(1, hop+1)]

    return pretrained_embs, mask

def eval_model(args):
    text_only = args.input_mode == "text_only"
    lp_diagnostics = getattr(args, 'lp_diagnostics', False)
    if text_only and args.task not in ("nctext", "lp"):
        raise ValueError("text_only requires nctext or LP diagnostics.")
    if args.task == 'lp' and (text_only or args.lp_ablation != 'full' or args.lp_context != 'official') and not lp_diagnostics:
        raise ValueError('LP input changes require --lp_diagnostics.')
    if lp_diagnostics and (args.task != 'lp' or args.template != 'HO' or args.dataset not in ('cora', 'pubmed', 'arxiv')):
        raise ValueError('LP diagnostics supports HO cora/pubmed/arxiv only.')
    if lp_diagnostics and args.lp_ablation == 'hop_mismatch' and args.lp_context == 'official' and args.dataset not in SMALL_DATASETS:
        raise ValueError('Official-cache mismatch is supported only for small graphs; use clean context for arxiv.')
    if text_only and args.lp_ablation != 'full':
        raise ValueError('Text-only does not accept graph ablations.')
    if args.prompt_style == "matched" and args.task not in ("nc", "nctext"):
        raise ValueError("matched prompts are only supported for nc/nctext.")
    # Model
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loaded from {model_path}. Model Base: {args.model_base}")
    if text_only:
        if not args.model_base:
            raise ValueError("text_only requires --model_base pointing to the frozen base LLM.")
        # Use the plain LLM: the LLaGA forward path always invokes the Projector.
        load_path, load_base, load_name = args.model_base, None, "text_only_base"
        print(f"Text-only: loading base LLM {load_path}; no Projector or graph features.")
    else:
        load_path, load_base, load_name = model_path, args.model_base, model_name
    tokenizer, model, context_len = load_pretrained_model(load_path, load_base, load_name,
                                                          # Evaluation below uses one CUDA device; avoid CPU offload.
                                                          device_map={"": torch.cuda.current_device()},
                                                          cache_dir=args.cache_dir)
    model = model.to(torch.float16).cuda()
    # data_dir=os.path.expanduser(args.data_dir)
    if args.dataset == "arxiv":
        data_dir = "dataset/ogbn-arxiv"
    elif args.dataset == "products":
        data_dir = "dataset/ogbn-products"
    elif args.dataset == "pubmed":
        data_dir = "dataset/pubmed"
    elif args.dataset == "cora":
        data_dir = "dataset/cora"
    else:
        print(f"{args.dataset} not exists")
        raise ValueError
    data_dir = os.path.join(args.data_root, os.path.basename(data_dir))
    if args.lp_context == 'clean_cached':
        if args.template != 'HO' or args.task != 'lp':
            raise ValueError('Clean cached context requires HO LP.')
        with open(os.path.join(data_dir, 'clean_lp_manifest.json')) as f:
            clean_manifest = json.load(f)
        if (clean_manifest.get('protocol') != 'disjoint_context_lp_v1' or clean_manifest.get('embedding') != args.pretrained_embedding_type
                or not clean_manifest.get('features_complete')):
            raise ValueError('Missing/incompatible clean LP manifest.')
    if args.task in  ["nc", "nd", "nda", "nctext"]:
        if args.template == "HO":
            prompt_file = os.path.join(data_dir, f"sampled_2_10_test.jsonl")
        else:
            prompt_file = os.path.join(data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_test.jsonl")
        data_path = os.path.join(data_dir, f"processed_data.pt")
    elif args.task in ["lp"]:
        if args.template == "HO":
            prompt_file = os.path.join(data_dir, f"edge_sampled_2_10_only_test.jsonl")
        else:
            prompt_file = os.path.join(data_dir, f"edge_sampled_{args.use_hop}_{args.sample_neighbor_size}_only_test.jsonl")
        data_path = os.path.join(data_dir, f"processed_data.pt")
    else:
        raise ValueError

    if args.task == 'lp' and args.lp_split != 'test':
        if not lp_diagnostics or args.lp_context != 'clean_cached':
            raise ValueError('Non-test LP split is supported only with clean cached diagnostics.')
        prompt_file = os.path.join(data_dir, f'edge_sampled_2_10_only_{args.lp_split}.jsonl')
    data = torch.load(data_path)
    print(f"Load from {prompt_file}\n")
    lines = open(prompt_file, "r").readlines()

    if args.start >= 0:
        if args.end < 0:
            args.end = len(lines)
        lines = lines[args.start:args.end]
    elif args.end > 0:
        lines = lines[:args.end]

    lp_run_config = None
    if lp_diagnostics:
        def digest_file(path):
            digest = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(1048576), b''):
                    digest.update(chunk)
            return digest.hexdigest()
        lp_run_config = dict(protocol='lp_input_screen_v1', dataset=args.dataset,
                             data_root=os.path.realpath(args.data_root), context=args.lp_context, split=args.lp_split,
                             input_mode=args.input_mode, ablation=args.lp_ablation,
                             ablation_seed=args.ablation_seed, greedy=args.greedy,
                             seed=args.seed, temperature=args.temperature,
                             max_new_tokens=args.max_new_tokens, text_max_chars=args.lp_text_max_chars,
                             use_hop=args.use_hop, embedding=args.pretrained_embedding_type,
                             model_path=os.path.realpath(args.model_base if text_only else args.model_path),
                             model_base=os.path.realpath(args.model_base),
                             prompt_sha256=digest_file(prompt_file),
                             code_sha256=digest_file(__file__),
                             input_code_sha256=digest_file(os.path.join(os.path.dirname(__file__), 'lp_inputs.py')),
                             start=args.start, end=args.end)
        if not text_only:
            lp_run_config['projector_sha256'] = digest_file(os.path.join(args.model_path, 'mm_projector.bin'))
        if args.lp_context == 'clean_cached':
            lp_run_config['manifest_sha256'] = digest_file(os.path.join(data_dir, 'clean_lp_manifest.json'))
        if os.path.exists(args.answers_file):
            previous = [json.loads(s) for s in open(args.answers_file) if s.strip()]
            if len(previous) > len(lines):
                raise ValueError('Existing output longer than selection.')
            for i, old in enumerate(previous):
                ref = json.loads(lines[i])
                if (old.get('lp_diagnostic_config') != lp_run_config or old['question_id'] != ref['id']
                        or old['gt'] != ref['conversations'][1]['value']):
                    raise ValueError('Different protocol/order in existing output; choose a new output path.')

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    if "tmp" not in args.answers_file and os.path.exists(answers_file):
        line_number = len(open(answers_file, 'r').readlines())
        print(f"{args.answers_file} already exists! it has {line_number} lines!!")
        if line_number >= len(lines):
            return
        lines = lines[line_number:]
        ans_file = open(answers_file, "a")
    else:
        ans_file = open(answers_file, "w")

    questions = [json.loads(q) for q in lines]

    index = None
    if text_only:
        pretrained_emb = structure_emb = None
    elif args.template == "ND":
        pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
        structure_emb = torch.load(
            f"dataset/laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt")

    elif args.template == "HO":
        n = data.num_nodes
        if args.task == 'lp' and args.lp_context == 'clean_cached':
            mask = torch.ones(n, dtype=torch.bool)
            pretrained_emb = load_pretrain_embedding_hop(data_dir, args.pretrained_embedding_type, args.use_hop, mask)
            index = torch.arange(n)
        elif args.dataset in SMALL_DATASETS and args.task == "lp":
            pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
        elif args.task == "lp":
            # for small dataset, we remove test link during testing
            # for large dataset, remove test link and compute embedding may be more memory- and time-consuming , we precompute the embedding
            pretrained_emb, mask = load_pretrain_embedding_hop_lp(data_dir, args.pretrained_embedding_type,args.use_hop)
            index = torch.full([n], fill_value=n + 1, dtype=torch.long)
            test_index = torch.arange(mask.sum())
            index[mask] = test_index
        else:
            mask = torch.full([n], fill_value=False, dtype=torch.bool)
            for q in questions:
                idx = q["id"]
                if "lp" in  args.task:
                    assert len(idx) == 2
                    mask[idx[0]] = True
                    mask[idx[1]] = True
                elif args.task  in ["nc", "nd", "nctext"]:
                    assert isinstance(idx, int)
                    mask[idx] = True
            pretrained_emb = load_pretrain_embedding_hop(data_dir, args.pretrained_embedding_type, args.use_hop, mask)
            index = torch.full([n], fill_value=n + 1, dtype=torch.long)
            test_index = torch.arange(mask.sum())
            index[mask] = test_index
        structure_emb = None
    else:
        raise ValueError


    for line in tqdm(questions):
        idx = line["id"]
        if args.task in ["nd", "nda"]:
            qs=f"Please briefly describe the center node of {DEFAULT_GRAPH_TOKEN}."
        elif args.task == "nc" and args.prompt_style == "official":
            if args.dataset == "products":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent products sold in Amazon, and edges between products indicate they are purchased together. We need to classify the center node into 47 classes: Home & Kitchen, Health & Personal Care, Beauty, Sports & Outdoors, Books, Patio, Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones & Accessories, Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, Electronics, Movies & TV, Software, Video Games, Automotive, Pet Supplies, Office Products, Industrial & Scientific, Musical Instruments, Tools & Home Improvement, Magazine Subscriptions, Baby Products, label 25, Appliances, Kitchen & Dining, Collectibles & Fine Art, All Beauty, Luxury Beauty, Amazon Fashion, Computers, All Electronics, Purchase Circles, MP3 Players & Accessories, Gift Cards, Office & School Supplies, Home Improvement, Camera & Photo, GPS & Navigation, Digital Music, Car Electronics, Baby, Kindle Store, Buy a Kindle, Furniture & D&#233;cor, #508510, please tell me which class the center node belongs to?"
            else:
                qs = line["conversations"][0]['value']
        elif args.task == "nctext" or (args.task == "nc" and args.prompt_style == "matched"):
            text = data.raw_texts[line['id']][:2000] if args.task == "nctext" else ""
            if args.dataset == "arxiv":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers and edges represent co-citations, the node feature of center node is {text}. We need to classify the center node into 40 classes: cs.NA(Numerical Analysis), cs.MM(Multimedia), cs.LO(Logic in Computer Science), cs.CY(Computers and Society), cs.CR(Cryptography and Security), cs.DC(Distributed, Parallel, and Cluster Computing), cs.HC(Human-Computer Interaction), cs.CE(Computational Engineering, Finance, and Science), cs.NI(Networking and Internet Architecture), cs.CC(Computational Complexity), cs.AI(Artificial Intelligence), cs.MA(Multiagent Systems), cs.GL(General Literature), cs.NE(Neural and Evolutionary Computing), cs.SC(Symbolic Computation), cs.AR(Hardware Architecture), cs.CV(Computer Vision and Pattern Recognition), cs.GR(Graphics), cs.ET(Emerging Technologies), cs.SY(Systems and Control), cs.CG(Computational Geometry), cs.OH(Other Computer Science), cs.PL(Programming Languages), cs.SE(Software Engineering), cs.LG(Machine Learning), cs.SD(Sound), cs.SI(Social and Information Networks), cs.RO(Robotics), cs.IT(Information Theory), cs.PF(Performance), cs.CL(Computational Complexity), cs.IR(Information Retrieval), cs.MS(Mathematical Software), cs.FL(Formal Languages and Automata Theory), cs.DS(Data Structures and Algorithms), cs.OS(Operating Systems), cs.GT(Computer Science and Game Theory), cs.DB(Databases), cs.DL(Digital Libraries), cs.DM(Discrete Mathematics), please tell me which class the center node belongs to? Direct tell me the class name."
            elif args.dataset == "products":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent products sold in Amazon, and edges between products indicate they are purchased together, the node feature of center node is {text}. We need to classify the center node into 47 classes: Home & Kitchen, Health & Personal Care, Beauty, Sports & Outdoors, Books, Patio, Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones & Accessories, Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, Electronics, Movies & TV, Software, Video Games, Automotive, Pet Supplies, Office Products, Industrial & Scientific, Musical Instruments, Tools & Home Improvement, Magazine Subscriptions, Baby Products, label 25, Appliances, Kitchen & Dining, Collectibles & Fine Art, All Beauty, Luxury Beauty, Amazon Fashion, Computers, All Electronics, Purchase Circles, MP3 Players & Accessories, Gift Cards, Office & School Supplies, Home Improvement, Camera & Photo, GPS & Navigation, Digital Music, Car Electronics, Baby, Kindle Store, Buy a Kindle, Furniture & D&#233;cor, #508510, please tell me which class the center node belongs to? Direct tell me the class name."
            elif args.dataset == "pubmed":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers about Diabetes and edges represent co-citations, the node feature of center node is {text}. We need to classify the center node into 3 classes: Diabetes Mellitus Experimental, Diabetes Mellitus Type1, Diabetes Mellitus Type2, please tell me which class the center node belongs to? Direct tell me the class name."
            elif args.dataset == "cora":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers and edges represent co-citations, the node feature of center node is {text}. We need to classify the center node into 7 classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory, please tell me which class the center node belongs to? Direct tell me the class name."
            else:
                raise ValueError
            if args.task == "nc":
                # Matched E uses the same class list/instruction as E+T, without raw text.
                qs = qs.replace("the node feature of center node is . ", "", 1)
            if text_only:
                # Keep exactly the text/classification suffix used by E+T.
                marker = "the node feature of center node is "
                qs = "Given a node, " + qs[qs.index(marker):]
                if DEFAULT_GRAPH_TOKEN in qs:
                    raise ValueError("Raw text contains a reserved graph token.")
        elif args.task == "lp":
            qs=f"Given two node-centered subgraphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, we need to predict whether these two nodes connect with each other. Please tell me whether two center nodes in the subgraphs should connect to each other."
            if text_only:
                qs = endpoint_prompt(data.raw_texts, idx, args.lp_text_max_chars)
        else:
            print(f"NOT SUPPORT {args.task}!!!")
            raise ValueError
        cur_prompt = qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_graph_token(prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        if not text_only and not isinstance(line['graph'][0], list):
            line['graph'] = [line['graph']]
        if text_only:
            graph = graph_emb = None
        elif args.template == "ND":
            graph = torch.LongTensor(line['graph'])
            mask = graph != DEFAULT_GRAPH_PAD_ID
            masked_graph_emb = pretrained_emb[graph[mask]]
            s, n, d = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
            graph_emb = torch.zeros((s, n, d))
            graph_emb[mask] = masked_graph_emb
            if structure_emb is not None:
                graph_emb = torch.cat([graph_emb, structure_emb.unsqueeze(0).expand(s, -1, -1)], dim=-1)
        elif args.template == "HO":
            # for small dataset, we remove test link during testing
            # for large dataset, remove test link and compute embedding may be more memory- and time-consuming , we precompute the embedding

            if args.dataset in SMALL_DATASETS and args.task == "lp" and args.lp_context == 'official':
                mp = MP()
                center_nodes = []
                for g in range(len(line['graph'])):
                    center_id = line['graph'][g][0]
                    line['graph'][g] = [center_id] * (args.use_hop + 1)
                    center_nodes.append(center_id)
                graph = torch.LongTensor(line['graph'])
                center_id = graph[:, 0]
                graph_embs = [pretrained_emb[center_id].cuda()]
                subset, edge_index, mapping, edge_mask = k_hop_subgraph(center_nodes, args.use_hop, data.edge_index,
                                                                        relabel_nodes=True)
                local_edge_mask = ((edge_index[0] == mapping[0]) & (edge_index[1] == mapping[1])) | (
                            (edge_index[0] == mapping[1]) & (edge_index[1] == mapping[0]))
                edge_index = edge_index[:, ~local_edge_mask]
                local_x = pretrained_emb[subset].cuda()
                n = subset.shape[0]
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index)
                edge_index = edge_index.cuda()
                row, col = edge_index
                deg = degree(col, n, dtype=pretrained_emb.dtype)
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                # local_x = pretrained_emb
                for _ in range(args.use_hop):
                    local_x = mp.propagate(edge_index, x=local_x, norm=norm)
                    graph_embs.append(local_x[mapping])
                graph_emb = torch.stack(graph_embs, dim=1)
            else:

                for g in range(len(line['graph'])):
                    center_id = line['graph'][g][0]
                    line['graph'][g] = [center_id]*(args.use_hop+1)
                graph = torch.LongTensor(line['graph'])
                center_id = graph[:, 0]
                graph_emb = torch.stack([emb[index[center_id]] for emb in pretrained_emb], dim=1)
        else:
            raise ValueError

        donor_ids = None
        if lp_diagnostics and not text_only:
            donor_emb = None
            if args.lp_ablation == 'hop_mismatch':
                donor_ids = donors_for_pair(idx, data.num_nodes, args.ablation_seed)
                if args.lp_context == 'official':
                    donor_emb = pair_ho(MP(), data.edge_index, pretrained_emb, donor_ids, idx, args.use_hop, 'cuda')
                else:
                    donor_emb = torch.stack([emb[donor_ids] for emb in pretrained_emb], dim=1)
            graph_emb = intervene(graph_emb, args.lp_ablation, donor_emb)
            graph = torch.tensor(idx, dtype=torch.long).unsqueeze(1).expand(-1, graph_emb.shape[1]).clone()
            line['graph'] = graph.tolist()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        try:
            if args.seed is not None:
                # Stable per node/pair, including when resuming an interrupted result file.
                seed_key = json.dumps([args.seed, args.dataset, idx], separators=(",", ":"))
                sample_seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:4], "big")
                set_seed(sample_seed)
            graph_kwargs = {} if text_only else {
                "graph_emb": graph_emb.half().cuda(), "graph": graph.cuda()}
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    **graph_kwargs,
                    do_sample=not args.greedy,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
        except Exception as e:
            if args.fail_on_error:
                ans_file.close()
                raise
            print(f"!!!!!!Error!!!!! {e}")
            outputs=""

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "graph": [] if text_only else line['graph'],
                                   "text": outputs,
                                   "gt":line["conversations"][1]['value'],
                                   "answer_id": ans_id,
                                   **({'lp_diagnostic_config': lp_run_config, 'donor_ids': donor_ids} if lp_diagnostics else {}),
                                   "eval_config": {
                                       "model_path": load_path, "model_base": load_base,
                                       "task": args.task, "input_mode": args.input_mode,
                                       "prompt_style": args.prompt_style, "template": args.template,
                                       "embedding": None if text_only else args.pretrained_embedding_type,
                                       "use_hop": None if text_only else args.use_hop,
                                       "seed": args.seed, "seed_policy": "per_sample_sha256" if args.seed is not None else None,
                                       "do_sample": not args.greedy, "temperature": args.temperature,
                                       "max_new_tokens": args.max_new_tokens,
                                       "text_max_chars": 2000 if args.task == "nctext" else None,
                                   }}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    # parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--pretrained_embedding_type", type=str, default="sbert")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=5)
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--mm_use_graph_start_end",default=False, action="store_true")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--template", type=str, default="ND")
    parser.add_argument("--input_mode", choices=["graph", "text_only"], default="graph")
    parser.add_argument("--prompt_style", choices=["official", "matched"], default="official")
    parser.add_argument("--seed", type=int, default=None, help="Optional per-sample inference seed; separate from training seed.")
    parser.add_argument("--greedy", action="store_true", help="Disable sampling; default retains official sampling.")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--fail_on_error", action="store_true")
    parser.add_argument('--data_root', default='dataset')
    parser.add_argument('--lp_diagnostics', action='store_true')
    parser.add_argument('--lp_ablation', choices=['full', 'center_only', 'center_repeat', 'hop_mismatch'], default='full')
    parser.add_argument('--lp_context', choices=['official', 'clean_cached'], default='official')
    parser.add_argument('--ablation_seed', type=int, default=42)
    parser.add_argument('--lp_text_max_chars', type=int, default=2000)
    parser.add_argument('--lp_split', choices=['train', 'val', 'test'], default='test')
    args = parser.parse_args()

    eval_model(args)
