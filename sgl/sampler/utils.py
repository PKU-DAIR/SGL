import numpy as np
import threading

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
            
class threadsafe_generator:  
    """Takes an generator and makes it thread-safe by 
    serializing call to the `next` method of given generator. 
    """  
    def __init__(self, gen):  
        self.gen = gen  
        self.lock = threading.Lock()  
  
    def __iter__(self):  
        return self 
  
    def __next__(self):  
        with self.lock:  
            return self.gen.__next__() 


class MiniBatch(object):
    def __init__(self, seed_nodes, batch_size):
        self.seed_nodes = seed_nodes
        self.batch_size = batch_size

    def __iter__(self):
        pass

    def __call__(self):
        pass
         
class RandomBatch(MiniBatch):
    def __init__(self, seed_nodes, batch_size):
        super().__init__(seed_nodes, batch_size)
        self.num_batches = (len(seed_nodes) + batch_size - 1) // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = np.random.choice(
                self.seed_nodes, self.batch_size, replace=False)
            yield batch

    def __call__(self):
        batch = np.random.choice(
                self.seed_nodes, self.batch_size, replace=False)
        
        return np.sort(batch)