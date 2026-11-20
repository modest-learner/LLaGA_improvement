from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from defusedxml import minidom
A='http://schemas.openxmlformats.org/drawingml/2006/main'
src=Path('LLaGA_跨图LP研究进展汇报_20261120.pptx'); dst=src.with_name(src.stem+'_宋体_TimesNewRoman.pptx')
changed=[]
with ZipFile(src) as zin,ZipFile(dst,'w',ZIP_DEFLATED) as zout:
 for item in zin.infolist():
  b=zin.read(item.filename)
  if item.filename.startswith('ppt/') and item.filename.endswith('.xml'):
   d=minidom.parseString(b)
   for name in ['latin','ea','cs']:
    for e in d.getElementsByTagNameNS(A,name):e.setAttribute('typeface','宋体' if name=='ea' else 'Times New Roman')
   for e in d.getElementsByTagNameNS(A,'font'):
    if e.getAttribute('script') in ['Hans','Hant']:e.setAttribute('typeface','宋体')
   for tag in ['rPr','defRPr','endParaRPr']:
    for e in d.getElementsByTagNameNS(A,tag):
     e.setAttribute('lang','en-US');e.setAttribute('altLang','zh-CN')
     # Typeface children follow fill/effect/highlight, precede hyperlink/rtl/ext.
     for n in list(e.childNodes):
      if n.nodeType==n.ELEMENT_NODE and n.namespaceURI==A and n.localName in ['latin','ea','cs','sym']:e.removeChild(n)
     before=next((n for n in e.childNodes if n.nodeType==n.ELEMENT_NODE and n.localName in ['hlinkClick','hlinkMouseOver','rtl','extLst']),None)
     for tag2,font in [('latin','Times New Roman'),('ea','宋体'),('cs','Times New Roman')]:
      n=d.createElementNS(A,'a:'+tag2);n.setAttribute('typeface',font)
      if before:e.insertBefore(n,before)
      else:e.appendChild(n)
   b=d.toxml(encoding='UTF-8');changed.append(item.filename)
  zout.writestr(item,b)
print(dst,'updated XML parts:',len(changed))
