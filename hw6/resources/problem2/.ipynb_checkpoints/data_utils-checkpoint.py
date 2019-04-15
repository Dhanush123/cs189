import glob, json
import numpy as np
import ipdb

def load_data(path,debug):
    if debug:
        print('Debug is ON!')
        load_key = 'gt_pose_align'
    else:
        load_key = 'gt_pose'
    jsons_train = {'squat':[],'reach':[],'lunge':[],'inline':[],
                   'hamstrings':[],'stretch':[],'deadbug':[],'pushup':[]
                  }
    jsons_val = {'squat':[],'reach':[],'lunge':[],'inline':[],
                 'hamstrings':[],'stretch':[],'deadbug':[],'pushup':[]
                }
    for (id,person) in enumerate(glob.glob(path+"/*/")):
        for move in glob.glob(person+"labels/*.json"):
            this = json.load(open(move,"r"))
            classid = move.split("/")[-1].split(".")[0]
            if id < 225:
                # NOTE: for bilateral movements, we grab the same example twice
                #  so then we don't have to worry about loss weighting
                #  due to class imbalance
                jsons_train[classid].append(np.array(this[load_key]['left']))
                jsons_train[classid].append(np.array(this[load_key]['right']))
            else:
                jsons_val[classid].append(np.array(this[load_key]['left']))
                jsons_val[classid].append(np.array(this[load_key]['right']))

    return jsons_train,jsons_val

def to_matrix(jsons):
    features = []
    labels = []
    sorted_key = sorted(jsons.keys())
    for counter, key in enumerate(sorted_key):
        for item in jsons[key]:
            features.append(item)
            labels.append(counter)
    features = np.array(features)
    labels = np.array(labels)
    return features, labels

def load_mds189(path,debug):
    js_train,js_val = load_data(path,debug)
    features_train, labels_train = to_matrix(js_train)
    feat_val, label_val = to_matrix(js_val)
    # random permute to shuffle the data
    np.random.seed(0)
    perm_train = np.random.permutation(features_train.shape[0])
    feat_train=features_train[perm_train, :]
    label_train=labels_train[perm_train]

    return feat_train, label_train, feat_val, label_val

# path = '/Users/panna/cs189/mds189/trainval'
# debug = False
# feat_train, label_train, feat_val, label_val = load_mds189(path,debug)
