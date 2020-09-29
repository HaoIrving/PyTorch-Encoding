import xml.etree.ElementTree as ET
import os
def __indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
             elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            __indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

root=ET.Element('annotation')
root.text='\n'
tree=ET.ElementTree(root)

#parameters to set
#filename=os.walk('/input_path')[2]
filename='1.tif'
resultfile=filename.split('.')[0]+'_gt.png'
resultfile_xml='./'+filename.split('.')[0]+'.xml'
organization='CASIA'
author='1,2,3,4,5,6'

element_source=ET.Element('source')
element_source.text='\n'+7*' '
element_source.tail='\n'+4*' '
element_filename=ET.Element('filename')
element_filename.tail='\n'+7*' '
element_filename.text=filename

element_origin=ET.Element('origin')
element_origin.tail='\n'+4*' '
element_origin.text='GF2/GF3'
element_research=ET.Element('research')
element_research.text='\n'+7*' '
element_research.tail='\n'+4*' '
element_version=ET.Element('version')
element_version.tail='\n'+7*' '
element_version.text='4.0'
element_provider=ET.Element('provider')
element_provider.tail='\n'+7*' '
element_provider.text=organization
element_author=ET.Element('author')
element_author.text=author
element_author.tail='\n'+7*' '
element_pluginname=ET.Element('pluginname')
element_pluginname.tail='\n'+7*' '
element_pluginname.text='地物标注'
element_pluginclass=ET.Element('pluginclass')
element_pluginclass.tail='\n'+7*' '
element_pluginclass.text='标注'
element_time=ET.Element('time')
element_time.tail='\n'+4*' '
element_time.text='2020-07-2020-11'
element_seg=ET.Element('segmentation')
element_seg.text='\n'+7*' '
element_seg.tail='\n'
element_resultfile=ET.Element('resultflie')
element_resultfile.tail='\n'+4*' '
element_resultfile.text=resultfile

#add
element_source.append(element_filename)
element_source.append(element_origin)

element_research.append(element_version)
element_research.append(element_provider)
element_research.append(element_author)
element_research.append(element_pluginname)
element_research.append(element_pluginclass)
element_research.append(element_time)

element_seg.append(element_resultfile)

root.append(element_source)
root.append(element_research)
root.append(element_seg)




#write
tree.write(resultfile_xml,encoding='utf-8',xml_declaration=True)

