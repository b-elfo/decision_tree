import numpy as np
import pandas as pd

from data import get_data

###

class Node:
    def __init__(self, 
                 node_idxs: list,
                 prev_depth: int = 0,
                 split_idx: int = 0,
                 split_vals: tuple = None,
                 is_terminal: bool = False,
                 ):
        self.split_idx = split_idx
        self.split_vals = split_vals
        self.node_idxs = node_idxs
        self.children = []
        self.depth = prev_depth+1
        self.is_terminal = is_terminal

###

class DecisionTree:
    def __init__(self,
                 data: pd.DataFrame,
                 goal_label: str, 
                 max_depth: int = 5,
                 min_feats: int = 10,
                 num_class: int = 0,
                 ):
        all_idxs = np.array(range(len(data)))
        self.root = Node(node_idxs=all_idxs)
        self.max_depth = max_depth
        
        self.min_feats = min_feats
        self.num_class = num_class

        self.goal_label = goal_label

        self.grow_tree(data=data,
                       )
    
    def grow_tree(self, 
                  data: pd.DataFrame,
                  ):
        # node_queue = [self.root]
        # while len(node_queue) > 0:
        self.split_node(node=self.root,
                        data=data,
                        )

    def split_node(self,
                   node: Node,
                   data: pd.DataFrame,
                   ):
        # print(node.split_idx, len(node.node_idxs))
        
        curr_node = node if node else self.root
        if curr_node.depth >= self.max_depth:
            curr_node.is_terminal = True
            # print('Reached maximum depth!')
            return
        
        if len(curr_node.node_idxs) < self.min_feats:
            curr_node.is_terminal = True
            # print('Not enough data features to split node!')
            return
            
        node_data = data.iloc[curr_node.node_idxs]
        
        split_col, split_vals = best_split(df=node_data,
                                           goal_label=self.goal_label,
                                           )
        
        # print(split_col, split_vals)
        for val0,val1 in zip(split_vals[:],np.append(split_vals[1:],split_vals[-1]+1e-6)):
            data_idxs = data[data[split_col].between(val0,val1,inclusive='left')].index.to_numpy()
            child = Node(split_idx=split_col,
                         split_vals=(val0,val1),
                         prev_depth=curr_node.depth,
                         node_idxs=data_idxs,
                         )
            curr_node.children.append(child)
            
            # child_data = data.iloc[data_idxs]
            self.split_node(child,
                            data.drop([split_col], axis=1), #child_data,
                            )
            # print('next child...')

    def predict(self, 
                X:pd.DataFrame,
                ):
        ###################
        # Still need this #
        ###################
        predictions = []

        print(self.root.children[0].split_idx, self.root.children[0].split_vals)
        print(self.root.children[1].split_idx, self.root.children[1].split_vals)

        for x in X:
            prediction = None
            predictions.append(prediction)
        
        return predictions
        
    def enumerate(self):
        nodes = [ self.root ]
        while len(nodes) > 0:
            node = nodes.pop(0)
            # print(f"{' '*node.depth}[{node.split_idx}] - [{node.split_vals}]")
            if not node.is_terminal:
                print(f"{' '*node.depth}[{node.split_idx}] - [{node.split_vals}]")
                for child in node.children[::-1]:
                    nodes.insert(0,child)
            else:
                print(f"{' '*node.depth}[{node.split_idx}] (terminal) - [{node.split_vals}]")

###

def best_split(df: pd.DataFrame,
               goal_label: str,
               ):
    best_col = ''
    best_ig = 0.
    best_split_vals = []
    for col in df.columns:
        if col not in [goal_label,'PassengerId','Cabin']:
            ig, split_vals = information_gain(df=df, split_idx=col, tar_idx=goal_label)
            # ig, split_vals = gini_index()
            if ig > best_ig:
                best_ig = ig
                best_col = col
                best_split_vals = split_vals
            # print(f"{col+' information gain:':35} {ig:.5f}")
    return best_col, np.sort(best_split_vals)

### Information Gain - Classification
def entropy(probs):
    entropy = 0.
    for prob in probs:
        entropy -= (prob)*np.log2(prob) if prob > 0 else 0.
    return entropy

def compute_probs(values: list,
                  tar_vals: list,
                  df: pd.DataFrame,
                  col_idx: str,
                  tar_idx: str,
                  ):
    counts = []
    for val in values:
        tar_counts = []
        for tar_val in tar_vals:
            tar_counts.append(len(df[(df[col_idx]==val) & (df[tar_idx]==tar_val)]))
        counts.append(tar_counts)
    probs = []
    for val_count in counts:
        probs.append([v / sum(val_count) for v in val_count])
    return probs, counts

def bucket_probs(tar_values: list,
                 df: pd.DataFrame,
                 col_idx: str,
                 num_buckets: int = 2,
                 ):
    # num_buckets == 2, average split
    data = df.dropna(subset=[col_idx])
    counts = []
    if num_buckets == 2:
        split_vals = [data[col_idx].min(),data[col_idx].mean()] # ,data[col_idx].max()
    else:
        split_vals = [] # split into n buckets
    for val0,val1 in zip(split_vals[:-1],split_vals[1:]):
        tar_counts = []
        for val in tar_values:
            tar_counts.append(len(data[data[col_idx].between(val0,val1)==val]))
        counts.append(tar_counts)
    probs = []
    for val_count in counts:
        probs.append([v / sum(val_count) for v in val_count])
    return probs, counts, split_vals

def information_gain(df: pd.DataFrame,
                     split_idx: int,
                     tar_idx: int,
                     max_split_nodes: int = 5,
                     ):
    data_num = df[tar_idx].count()
    # parent entropy
    tar_vals = df[tar_idx].unique()
    if len(tar_vals) <= max_split_nodes:
        counts = []
        for val in tar_vals:
            tar_probs = counts.append(len(df[df[tar_idx]==val]))
        tar_probs = counts / data_num
    else: # for regression, we can take the average or bucket  
        pass
    parent_entropy = entropy(tar_probs) # find int to column map in pandas
    # compute split entropies
    split_vals = df[split_idx].unique()
    if len(split_vals) <= max_split_nodes:
        # print(f'raw count for {split_idx}...')
        split_probs, split_counts = compute_probs(values=split_vals,
                                                  tar_vals=tar_vals,  
                                                  df=df, 
                                                  col_idx=split_idx,
                                                  tar_idx=tar_idx,
                                                  )
    else: # for regression, we can take the average or bucket for 'val'
        # print(f'bucket count for {split_idx}...')
        split_probs, split_counts, split_vals = bucket_probs(tar_values=tar_vals,
                                                             df=df,
                                                             col_idx=split_idx,
                                                             num_buckets=2,
                                                             )
    split_entropy = 0.
    for idx, prob in enumerate(split_probs):
        split_entropy += entropy(prob) * (sum(split_counts[idx]) / data_num)
    # information gain
    # print(f'#####\nparent entropy: {parent_entropy}\nsplit entropy: {split_entropy}')
    return parent_entropy - split_entropy, split_vals

###

def get_unique_values(df: pd.DataFrame):
    unique_vals = {}
    for col in df.columns:
        if col == 'id':
            continue
        col_df = df[col]
        unique_vals[col] = col_df.unique()
    return unique_vals

def get_class_idx_maps(unique_vals: dict):
    class_to_idx = {}
    idx_to_class = {}
    for key in unique_vals:
        if isinstance(unique_vals[key][0], str):
            num_classes = len(unique_vals[key])
            class_to_idx[key] = {unique_vals[key][idx]:idx for idx in range(num_classes)}
            idx_to_class[key] = {idx: unique_vals[key][idx] for idx in range(num_classes)}
    return class_to_idx, idx_to_class

###

def sanity_check(X: pd.DataFrame,
                 Y: pd.DataFrame,
                 criterion: str="entropy",
                 splitter: str="best",
                 max_depth: int=5,
                 min_samples: int=2,
                 ):
    from sklearn import tree
    from sklearn.metrics import log_loss
    classifier = tree.DecisionTreeClassifier(criterion=criterion,splitter=splitter,max_depth=max_depth,min_samples_leaf=min_samples)

    # issue with this line ???
    classifier = classifier.fit(X,Y)
    ###
    
    x1 = 0
    x2 = x1+10
    predictions = classifier.predict(X.iloc[x1:x2])
    truths = Y.iloc[x1:x2].values

    loss = log_loss(y_true=[truth[0] for truth in truths],
                    y_pred=predictions,
                    )
    
    text_repr = tree.export_text(classifier)
    print(text_repr)
    print(f"===\nLoss: {loss}")

###

if __name__=='__main__':
    df = get_data('titanic/train.csv')
    df = df.drop(labels=['Name','Ticket','PassengerId'], axis=1)
    goal_label = 'Survived'

    unique_vals = get_unique_values(df)

    input_data = df[[key for key in unique_vals.keys() if key != goal_label]].copy()
    labels = df[[goal_label]].copy

    class_to_idx, idx_to_class = get_class_idx_maps(unique_vals)
    for key in class_to_idx:
        df[key] = df[key].map(class_to_idx[key])
    
    print("=========================\nBuilding decision tree...\n===")
    dt = DecisionTree(data=df,
                      goal_label=goal_label,
                      max_depth=4,
                      min_feats=10,
                      )

    dt.enumerate()

    dt.predict(df.iloc[0])

    print("=========================\n")

    # ### Sanity check
    # print("===============\nSanity check...\n===")
    # data = df[['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    # data = data.dropna()
    # X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    # Y = data[['Survived']]
    # X = X.replace(class_to_idx)
    # sanity_check(X=X,Y=Y,min_samples=10)