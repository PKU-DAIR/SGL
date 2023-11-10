import numpy as np

def dot_product_ratio(ori_neighbors, good_neighbors):
    s = np.sum(np.dot(ori_neighbors, good_neighbors))
    max_s = np.sum(np.power(ori_neighbors, 2))
    return s / max_s

def adj_train_analysis(adj, minimum_neighbors, similarity_threshold, step=1, low_quality_score=0.2):
    nodes_num = adj.get_shape()[0]
    sample_mark = []
    
    for i in range(nodes_num):
        adj_coo = adj.getrow(i).tocoo()
        neighbors = adj_coo.col.reshape(-1) 
        if len(neighbors) < minimum_neighbors:
            sample_mark.append(low_quality_score)
            continue
        else:
            avg = int(neighbors.mean())
            neighbors_length = len(neighbors)         
            if neighbors_length % 2 == 0: 
                good_neighbors = np.arange((avg-neighbors_length//2*step+step), (avg+neighbors_length//2*step+1*step), step, int)
            else:
                good_neighbors = np.arange((avg-neighbors_length//2*step+step), (avg+neighbors_length//2*step+2*step), step, int)
            
            similarity = dot_product_ratio(neighbors, good_neighbors)           
            if similarity > similarity_threshold:
                sample_mark.append(1)
            else:
                sample_mark.append(low_quality_score)
    
    sample_mark_np = np.asarray(sample_mark)
    
    return sample_mark_np
            