from glob import glob
import shutil
import re
import os

whitespace = re.compile(r'\s+')

import lxml.etree

# class legend:
#  - inside: 0
#  - beginning of new sentence (and, thus, also beginning of word): 1
#  - beginning of new token: 2

data_dir = 'data/eltec'
gt_dir = 'data/eltec_gt'

try:
    shutil.rmtree(gt_dir)
except FileNotFoundError:
    pass
os.mkdir(gt_dir)

languages = sorted(set(os.listdir(f'{data_dir}/orig/')))
#languages = ['eng']

for language in languages:
    os.mkdir(f'{gt_dir}/{language}')

    for fn in sorted(glob(f'{data_dir}/orig/{language}/*.xml')):
        orig_p = {}
        try:
            orig_tree = lxml.etree.parse(fn)
        except lxml.etree.XMLSyntaxError:
            print(f'invalid XML in: {fn}')
            continue
        for p in orig_tree.iterfind('//p'):
            n = p.attrib['n']
            t = ''.join(p.itertext())
            t = ' '.join(t.split())
            orig_p[n] = t

        segm_fn = f'{data_dir}/tokenized/{language}/xml/{os.path.basename(fn)}'
        try:
            segm_tree = lxml.etree.parse(segm_fn)
        except lxml.etree.XMLSyntaxError:
            print(f'invalid XML in: {fn}')


        all_chars, all_labels = [], []
        
        for p in segm_tree.iterfind('//p'):
            try:
                n = p.attrib['n']
                if n not in orig_p:
                    print(f'Incompatible XML identifiers in: {fn}')
                    continue
                
                segm_labels = []
                for s_idx, s in enumerate(p.iterfind('s')):
                    if not len(s):
                        continue
                    segm_labels.append('<S/>')
                    for w_idx, w in enumerate(s.iterfind('w')):
                        if w_idx != 0:
                            segm_labels.append('<W/>')
                        chars = ''.join(w.itertext()).replace(' ', '')
                        segm_labels.extend(chars)

                text, labels = list(orig_p[n]), []
                
                for char_idx, char in enumerate(text):
                    if char == ' ':
                        labels.append(0)
                    elif segm_labels[0] == char:
                        segm_labels.pop(0)
                        labels.append(0)
                    elif segm_labels[0] in ('<S/>', '<W/>'):
                        if segm_labels[0] == '<S/>':
                            labels.append(1)
                        elif segm_labels[0] == '<W/>':
                            labels.append(2)
                        while segm_labels[0] in ('<S/>', '<W/>'):
                            segm_labels.pop(0)
                        segm_labels.pop(0)
            except IndexError:
                continue

            # mark paragraph breaks with spaces:
            text += ' '
            labels.append(0)

            if len(text) == len(labels):
                all_chars.extend(text)
                all_labels.extend(labels)
            else:
                print(f'-> issue parsing #{n} in {fn}')
                #for c, l in zip(text, labels):
                #    print(c, '   ', l)
                #print(len(text), len(labels))
        
        new_fn = f'{gt_dir}/{language}/{os.path.basename(fn)}'.replace('.xml', '.tsv')
        with open(new_fn, 'w') as f:
            for c, l in zip(all_chars, all_labels):
                f.write('\t'.join((c, str(l)))+'\n')
