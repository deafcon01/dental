import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
#import pdb; pdb.set_trace();

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '../annotations/*.xml'):
        print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.find('object'):
            print('found object')
            for member in root.findall('object'):
                #fname = root.find('filename').text
                #fname.replace(' ','_')
                value = (root.find('filename').text,
                        int(root.find('size')[0].text),
                        int(root.find('size')[1].text),
                        member[0].text,
                        int(member[4][0].text),
                        int(member[4][1].text),
                        int(member[4][2].text),
                        int(member[4][3].text)
                        ) 
                xml_list.append(value)
        elif root.find('annotations'):
            filename= xml_file.split('.xml')[0]+'.jpg'
            filename = filename.replace("../annotations/","")
            im = Image.open(os.path.join(filename))
            w,h=im.size
            im.close()
            for member in root.findall('annotations'):
                #fname = root.find('meta/imageid').text
                #print(fname)
                #fname.replace(' ','_')
                value = (root.find('meta/imageid').text,
                        int(w),
                        int(h),
                        member[1].text,
                        float(member[3].text),
                        float(member[5].text),
                        float(member[4].text),
                        float(member[0].text)
                        )
                xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

import pdb
pdb.set_trace()
def main():
    for directory in ['test']:
        image_path = os.path.join(os.getcwd(), '{}/images'.format(directory))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('test_labels.csv', index=None)
        print('Successfully converted xml to csv.')


main()
