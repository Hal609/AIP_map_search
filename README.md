# TODO
## 1. Implement All Search Algorithms

| Algorithm      | Progress      |
| ------------- | ------------- |
| Breadth-First Search (**BFS**) | ✅ |
| Depth-First Search (**DFS**) | ✅ |
| Uniform Cost Search (**UCS**) | ✅ |
| A Search (**A\***) | ✅ |

## 2. Search

- Search: Use a closed set to track visited nodes and avoid revisiting. ✅
- If a node is already visited, check if a faster path to the node has been found and update accordingly.
- Record the following for each version:
    - Path found ✅
    - Time elapsed ✅
    - Path cost ✅


##  3. Path Visualisation ✅
- Search exploration (e.g., paths explored). 
- The final path found (different colors for clarity). 
- e.g.:
    - Yellow lines: Path exploration history. 
    - Red: Start state. 
    - Magenta: Goal state. 
    - Use Matplotlib to display results on the grid. 


## 4. Compare Algorithms

- Run several searches (without visualisation) and log: the time ✅ and memory ✅ used. 
- Graph the time and space complexity of the algorithms ✅
- Create a comparison table with the following metrics:
    - Time taken to compute the path. ✅
    - Path length (number of moves). ✅
    - Path cost. ✅
    - Memory/storage usage. ✅


| Algorithm |  Time (ms)  | Nodes Visited | Path Length (m) | Final Path Cost (H:m:s) | Memory Use (KB) |
| --------- | ----------- | ------------- | --------------- |------------------------ | --------------- |
| BFS       | 95.604      |   31,935      |     3,307       |      00:05:38.446       |      4,111.6    |
| DFS       | 2.23        |      4848     |    93,369       |      02:26:55.137       |      925.2      |
| UCS       |   17.99     |     11,528    |     3,421       |      00:04:59.311       |     3,235.2     |
| A*        | 6.317       |     6161      |     3,421       |      00:04:59.311       |      2,635.6    |

## 5. Code Formatting and Structure

- Fully comment and document the code. ✅
- Describe the function of the algorithms in docstrings. ✅
- Make the visuals pretty. ✅
- Write a proper README.
