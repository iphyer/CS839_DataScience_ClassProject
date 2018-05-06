
# coding: utf-8

# # Stage 4 Data Analysis on the Integrated Table

# First, we repeat what we did in stage 3 and get a set of matching tuples.


import pandas as pd
import py_entitymatching as em
A = em.read_csv_metadata('./TableA_lower.csv', key = 'ID')
B = em.read_csv_metadata('./TableB_lower.csv', key = 'ID')

block_f = em.get_features_for_blocking(A, B)
block_t = em.get_tokenizers_for_blocking()
block_s = em.get_sim_funs_for_blocking()
r = em.get_feature_fn('jaccard(dlm_dc0(ltuple["Title"]), dlm_dc0(rtuple["Title"]))', block_t, block_s)
em.add_feature(block_f, 'Title_Title_jac_dlm_dc0_dlm_dc0', r)

ob = em.OverlapBlocker()
C = ob.block_tables(A, B, 'Author', 'Author', 
                    l_output_attrs=['Title','Author','Publication','Format','ISBN','Series', 'Physical Details'], 
                    r_output_attrs=['Title','Author','Publication','Format','ISBN','Series', 'Physical Details'], 
                    overlap_size = 2)

D = ob.block_candset(C, 'Title', 'Title', overlap_size = 4)
label_S = pd.read_csv('./Set_G.csv')
# em.copy_properties(S, label_S)
em.set_property(label_S, 'key', '_id')
em.set_property(label_S, 'fk_ltable', 'ltable_ID')
em.set_property(label_S, 'fk_rtable', 'rtable_ID')
label_S_rtable = em.read_csv_metadata('./label_S_rtable.csv')
label_S_ltable = em.read_csv_metadata('./label_S_ltable.csv')
em.set_property(label_S, 'rtable', label_S_rtable)
em.set_property(label_S, 'ltable', label_S_ltable)
label_S['ltable_Title'] = label_S['ltable_Title'].apply(lambda x: x.lower())
label_S['rtable_Title'] = label_S['rtable_Title'].apply(lambda x: x.lower())
label_S['ltable_Author'] = label_S['ltable_Author'].apply(lambda x: x.lower())
label_S['rtable_Author'] = label_S['rtable_Author'].apply(lambda x: x.lower())
label_S['ltable_Publication'] = label_S['ltable_Publication'].apply(lambda x: x.lower())
label_S['rtable_Publication'] = label_S['rtable_Publication'].apply(lambda x: x.lower())
label_S['ltable_Format'] = label_S['ltable_Format'].apply(lambda x: x.lower())
label_S['rtable_Format'] = label_S['rtable_Format'].apply(lambda x: x.lower())
label_S['ltable_Series'] = label_S['ltable_Series'].apply(lambda x: x.lower())
label_S['rtable_Series'] = label_S['rtable_Series'].apply(lambda x: x.lower())
match_f = em.get_features_for_matching(A, B)
match_t = em.get_tokenizers_for_matching()
match_s = em.get_sim_funs_for_matching()
f1 = em.get_feature_fn('jaccard(dlm_dc0(ltuple["Title"]), dlm_dc0(rtuple["Title"]))', match_t, match_s)
# f2 = em.get_feature_fn('jaccard(dlm_dc0(ltuple["Author"]), dlm_dc0(rtuple["Author"]))', match_t, match_s)
f3 = em.get_feature_fn('jaccard(dlm_dc0(ltuple["Publication"]), dlm_dc0(rtuple["Publication"]))', match_t, match_s)
f4 = em.get_feature_fn('jaccard(dlm_dc0(ltuple["Series"]), dlm_dc0(rtuple["Series"]))', match_t, match_s)
em.add_feature(match_f, 'Title_Title_jac_dlm_dc0_dlm_dc0', f1)
# em.add_feature(match_f, 'Author_Author_jac_dlm_dc0_dlm_dc0', f2)
em.add_feature(match_f, 'Publication_Publication_jac_dlm_dc0_dlm_dc0', f3)
em.add_feature(match_f, 'Series_Series_jac_dlm_dc0_dlm_dc0', f4)


# Add blackbox feature

import re
# for Roman numerals matching
def Title_Title_blackbox_1(x, y):
    
    # get name attribute
    x_title = x['Title']
    y_title = y['Title']
#     regex_roman = '\s+[MDCLXVI]+\s+'
    regex_roman = '\s+[mdclxvi]+($|\s+)'
    x_match = None
    y_match = None
    if re.search(regex_roman, x_title):
        x_match = re.search(regex_roman, x_title).group(0).strip()
    if re.search(regex_roman, y_title):
        y_match = re.search(regex_roman, y_title).group(0).strip()

    if x_match is None or y_match is None:
        return False
    else:
        return x_match == y_match

em.add_blackbox_feature(match_f, 'blackbox_1', Title_Title_blackbox_1)


# for number matching (e.g. 6th edition)
def Title_Title_blackbox_2(x, y):
    # x, y will be of type pandas series
    
    x_title = x['Title']
    y_title = y['Title']
    regex_number = '\s+(\d+)\s*th'
    x_match = None
    y_match = None
    if re.search(regex_number, x_title):
        x_match = re.search(regex_number, x_title).group(1)
    if re.search(regex_number, y_title):
        y_match = re.search(regex_number, y_title).group(1)

    if x_match is None or y_match is None:
        return False
    else:
        return x_match == y_match

em.add_blackbox_feature(match_f, 'blackbox_2', Title_Title_blackbox_2)

# for number matching (e.g. 6th edition)
from fuzzywuzzy import fuzz
def Author_Author_blackbox_3(x, y):
    # x, y will be of type pandas series
    
    x_author = x['Author']
    y_author = y['Author']
    return fuzz.token_set_ratio(x_author, y_author)/100.0
    
em.add_blackbox_feature(match_f, 'blackbox_3', Author_Author_blackbox_3)
match_f = match_f[(match_f['left_attribute'] != 'ID') & (match_f['left_attribute'] != 'ISBN')]
match_f = match_f[(match_f['left_attribute'] != 'Format') & (match_f['left_attribute'] != 'Series')]
H = em.extract_feature_vecs(label_S, feature_table=match_f, attrs_after=['label'])
# RF
rf = em.RFMatcher(n_estimators = 300,max_depth = 300, name='RF')
rf.fit(table=H, 
       exclude_attrs=['_id', 'ltable_ID', 'rtable_ID', 'label'], 
       target_attr='label')
H_test = em.extract_feature_vecs(D, feature_table=match_f)
pred_table = rf.predict(table= H_test, 
                        exclude_attrs=['_id', 'ltable_ID', 'rtable_ID'], 
                        target_attr='predicted_labels', 
                        return_probs=True, 
                        probs_attr='proba', 
                        append=True)
# eval_summary = em.eval_matches(pred_table, 'label', 'predicted_labels')
# eval_summary


# In[14]:


pred_rows = pred_table[pred_table['predicted_labels'] == 1]


# ## Merge the tables
# 
# Here we obtained the matching set based on the prediction.
# 
# 
# There are 1337 matched books in total.

matched_set = D[D['_id'].isin(pred_rows['_id'])]
matched_set
matched_set.shape
matched_id = matched_set[['ltable_ID', 'rtable_ID']]


# We put the id of matching tuples in to dictionaries.

# AB_list=[['ltable_ID','rtable_ID'] for x in pred_table if predicted_labels==1]
AB_list = matched_id.values.tolist()
AB_dict =  {item[0]: item[1] for item in AB_list}
BA_dict =  {item[1]: item[0] for item in AB_list}

len(AB_dict.keys())


# Here we merge the matching tuples.
# 
# For each field of a matching tuple, we simply keep the one with longer string length.

df = matched_set.iloc[:, 3:10]

df = df.rename(columns={'ltable_Title' : 'Title',
                  'ltable_Author' : 'Author',
                   'ltable_Publication' : 'Publication',
                   'ltable_Format' : 'Format',
                   'ltable_ISBN' : 'ISBN',
                   'ltable_Series' : 'Series',
                   'ltable_Physical Details' : 'Physical Details'
                  })

attr=['Title','Author','Publication','Format','ISBN','Series', 'Physical Details']
i = 0
for index, r in matched_set.iterrows():
    for  a in attr:
        left_name = 'ltable_' + a
        right_name = 'rtable_' + a
#         print(index,' ' ,attr.index(a))
        if len(str(r[left_name])) < len(str(r[right_name])):
            
            df.iloc[i, attr.index(a)] = r[right_name]
        else:
            df.iloc[i, attr.index(a)] = r[left_name]
    i += 1


# We added an Source attribute to indicate the source of a tuple.

df['Source'] = 'ab'
A['Source'] = 'a'
B['Source'] = 'b'


# After merging the matching set, we concat the set with the original table A and B.

set_merged = pd.concat([df, A[~A['ID'].isin(AB_dict.keys())].iloc[:,1:], B[~B['ID'].isin(BA_dict.keys())].iloc[:,1:]])
A[~A['ID'].isin(AB_dict.keys())].shape
B[~B['ID'].isin(BA_dict.keys())].shape
set_merged


# Final merged set contains 9350 tuples in total.

# The final set contains 9350 tuples in total.
set_merged.shape

set_merged[set_merged['Source'] == 'a'].shape


# We save the set to Table_E.csv.
set_merged.to_csv('Table_E.csv', sep=',')

