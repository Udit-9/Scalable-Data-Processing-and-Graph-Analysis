import heapq
from threading import*
import numpy as np
import math
import sys

#Priority Queue class using heapq 
class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.lock = Lock()

    def push(self, item):
        with self.lock:
            heapq.heappush(self.queue, item)

    def pop(self):
        with self.lock:
            if self.queue:
                return heapq.heappop(self.queue)
            return None

# Function to calculate magnitude of a vector
def norm(v):
    length_of_vector = len(v)
    result = 0
    for i in range(length_of_vector):
        result += v[i]**2
    return math.sqrt(result)

# Function to calculate cosine similarity between two vectors
def cosine_similarity(v1, v2):
    length_of_vector = len(v1)
    result = 0
    if(norm(v1)!=0 and norm(v2)!=0):
        for i in range(length_of_vector):
            result += (v1[i])*(v2[i])
        result = result/(norm(v1)*norm(v2))
    return result

# Worker function for calculating similarity and pushing to the priority queue
def calculate_similarity(query_vector, items_data, priority_queue):
    for item_id, item_vector in items_data.items():
        similarity_score = cosine_similarity(query_vector[0], item_vector)
        priority_queue.push((-similarity_score, item_id))

#Function to find top k similar items
def find_most_similar_items(query_vector, items_data, k, num_threads):
    # priority queue that stores the top k elements of each thread
    priority_queue = PriorityQueue() 
    items_per_thread = len(items_data) // num_threads #data size of each thread
    # Start worker threads
    threads = []

    for i in range(num_threads):
        start_idx = i * items_per_thread 
        end_idx = (i + 1) * items_per_thread if i < num_threads - 1 else len(items_data)
        items_data_chunk = {item_id: item_vector for item_id, item_vector in list(items_data.items())[start_idx:end_idx]}
        
        thread = Thread(target=calculate_similarity, args=(query_vector, items_data_chunk, priority_queue))
        thread.start()
        threads.append(thread)

    # Wait for threads to finish
    for thread in threads:
        thread.join()

    # Get results from the priority queue
    most_similar_items = []
    for _ in range(k):
        item = priority_queue.pop()
        if item:
            most_similar_items.append((item[1], -item[0]))

    return most_similar_items

def read_items_from_file(file_path):
    items_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Split the item into individual values
            parts = line.strip().split()
            # The first value is the ID, and the rest are vector components
            item_id = parts[0]
            item_vector = np.array([float(x) for x in parts[1:]])
            items_data[item_id] = item_vector
    return items_data

def read_vectors_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Split the item into individual values and there is no ID here
    vectors = [list(map(float, line.strip().split())) for line in lines]
    return np.array(vectors)

# Run the program with command line arguments
def main():
    file_path = sys.argv[1]  
    query_path = sys.argv[2]
    k = int(sys.argv[4])  # Desired number of similar items
    num_threads = int(sys.argv[3])  # Number of threads

    # file_path = 'dataa.txt'  # Replace with the actual path to your file
    # query_path='query.txt'
    # k = 3  # Replace with the desired number of similar items
    # num_threads = 2  # Replace with the number of threads

    query_vector = read_vectors_from_file(query_path)  # Replace with your actual query vector
    items_data = read_items_from_file(file_path)


    # dt=np.array(items_data)
    # print(items_data)

    # print(query_vector)
    # print(items_data)


    most_similar_items = find_most_similar_items(query_vector, items_data, k, num_threads)

    # Printing the top k similar  items
    for item_id, similarity_score in most_similar_items:
        print(f"{item_id} {similarity_score}")

if __name__ == "__main__":
    main() # calling the main function
