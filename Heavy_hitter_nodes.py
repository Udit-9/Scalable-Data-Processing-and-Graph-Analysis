import pyspark
from pyspark import SparkContext
import sys
import math

def count_unique_heavy_hitters_and_triangles(input_file_path):
    sc = SparkContext(appName="UniqueHeavyHittersAndTriangles")

    def parse_edge(line):
        vertex_info = line.strip().split(" ")
        return (int(vertex_info[0]), int(vertex_info[1]))

    def calculate_unique_heavy_hitters(edges):
        total_edges = edges.count()
        sqrt_total_edges = math.sqrt(total_edges)
        
        vertex_degrees = edges.flatMap(lambda edge: [(edge[0], 1), (edge[1], 1)]) \
                              .reduceByKey(lambda a, b: a + b)
        
        unique_heavy_hitters_count = vertex_degrees.filter(lambda x: x[1] >= sqrt_total_edges).count()
        
        return unique_heavy_hitters_count

    def calculate_unique_triangles(edges):
        def preprocess_edges(edge):
            if edge[0] < edge[1]:
                return (edge[0], edge[1])
            else:
                return (edge[1], edge[0])

        edges = edges.map(preprocess_edges)

        def mapper1(edge):
            if edge[0] < edge[1]:
                return (edge[0], [edge[1]])
            else:
                return (edge[1], [edge[0]])
        
        output_map1 = (edges.map(mapper1).filter(lambda x: x is not None)
                            .reduceByKey(lambda x, y: x + y))

        def reducer1(item):
            unique_triangles = []
            for i in range(0, len(item[1])):
                for j in range(i + 1, len(item[1])):
                    unique_triangles.append(((item[1][i], item[1][j]), [item[0]]))
            return unique_triangles

        output_reducer1 = output_map1.flatMap(reducer1)
        output_reducer2 = edges.map(lambda edge: ((edge[0], edge[1]), ["*"]))
        output_reducer2 = output_reducer2.union(output_reducer1)
        output = output_reducer2.reduceByKey(lambda x, y: x + y).collect()

        def generate_triplets(data):
            unique_triplets = []
            for item in data:
                vertex_list = item[1]
                if "*" in vertex_list and len(vertex_list) != 1:
                    vertex_list = set(vertex_list) - {"*"}
                    for vertex in vertex_list:
                        unique_triplets.append((item[0][0], item[0][1], vertex))

            return unique_triplets

        return len(generate_triplets(output))

    graph_data = sc.textFile(input_file_path).map(parse_edge)

    unique_heavy_hitters = calculate_unique_heavy_hitters(graph_data)

    unique_triangles = calculate_unique_triangles(graph_data)

    sc.stop()

    return unique_heavy_hitters, unique_triangles

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path-to-file>")
        sys.exit(1)

    input_file_path = sys.argv[1]

    unique_heavy_hitters, unique_triangles = count_unique_heavy_hitters_and_triangles(input_file_path)

    print("No. of unique heavy hitter nodes:", unique_heavy_hitters)
    print("No. of unique triangles:", unique_triangles)
