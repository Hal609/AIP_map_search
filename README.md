# <ins>TODO</ins>
## 1. Implement All Search Algorithms

| Algorithm      | Progress      |
| ------------- | ------------- |
| Breadth-First Search (**BFS**) | ✅ |
| Depth-First Search (**DFS**) | ✅ |
| Uniform Cost Search (**UCS**) | TODO |
| A Search (**A\***) | TODO |

## 2. Tree Search and Graph Search

- Implement two paradigms for each algorithm:
- Tree Search: Explore all nodes without checking if a node has been visited before.
- Graph Search: Use a closed set to track visited nodes and avoid revisiting.
- Record the following for each version:
- Path found
- Time elapsed
- Path cost


##  3. Path Visualisation
- Search exploration (e.g., paths explored).
- The final path found (different colors for clarity).
- e.g.:
    - Yellow lines: Path exploration history.
    - Magenta line: Final path.
    - Red: Start state.
    - Green: Goal state.
    - Use Matplotlib to display results on the grid.


## 4. Compare Algorithms

- Run several searches (without visualisation) and log the time and memory used.
- Graph the time and space complexity of the algorithms 
- Create a comparison table with the following metrics:
    - Time taken to compute the path.
    - Path length (number of moves).
    - Path cost.
    - Memory/storage usage.

| Algorithm | Time (s) | Path Length | Path Cost | Memory Usage | 
| --------- | -------- | ----------- | --------- |------------- |
| BFS       |          |             |           |              | 
| DFS       |          |             |           |              | 
| UCS       |          |             |           |              | 
| A*        |          |             |           |              | 


## 5. Code Formatting and Structure

- Fully comment and document the code.
- Describe the function of the algorithms in docstrings.
- Make the visuals pretty.
- Write a proper README.