import random
from flask import Flask, render_template, request, jsonify
import csv
from collections import deque
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
import os
from datetime import datetime

app = Flask(__name__)

# Load words from CSV
def load_words_from_csv(filename):
    words = set()
    with open(filename, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                words.add(row[0].strip().lower())
    return words

valid_words = load_words_from_csv('final.csv')

# Create a subset of words by length for faster access
words_by_length = {}
for word in valid_words:
    length = len(word)
    if length not in words_by_length:
        words_by_length[length] = []
    words_by_length[length].append(word)

# Banned words for Challenge Mode
banned_words = set(random.sample(list(valid_words), 100))  # Randomly select 100 words to ban

# Search algorithms
def bfs(start_word, end_word):
    """Breadth-First Search to find a valid path."""
    if start_word == end_word:
        return [start_word]

    queue = deque([[start_word]])
    visited = set([start_word])

    while queue:
        path = queue.popleft()
        current_word = path[-1]

        for neighbor in get_neighbors(current_word):
            if neighbor == end_word:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])

    return []

def ucs(start_word, end_word):
    """Uniform Cost Search to find the shortest path."""
    if start_word == end_word:
        return [start_word]

    priority_queue = []
    heapq.heappush(priority_queue, (0, [start_word]))
    visited = set([start_word])

    while priority_queue:
        cost, path = heapq.heappop(priority_queue)
        current_word = path[-1]

        for neighbor in get_neighbors(current_word):
            if neighbor == end_word:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                heapq.heappush(priority_queue, (cost + 1, path + [neighbor]))

    return []

def a_star(start_word, end_word):
    """A* Search to find the shortest path."""
    def heuristic(word):
        return sum(1 for a, b in zip(word, end_word) if a != b)

    if start_word == end_word:
        return [start_word]

    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start_word), 0, [start_word]))
    visited = set([start_word])

    while open_list:
        _, cost, path = heapq.heappop(open_list)
        current_word = path[-1]

        if current_word == end_word:
            return path

        for neighbor in get_neighbors(current_word):
            if neighbor not in visited:
                visited.add(neighbor)
                f_cost = cost + 1 + heuristic(neighbor)
                heapq.heappush(open_list, (f_cost, cost + 1, path + [neighbor]))

    return []

def greedy_bfs(start_word, end_word):
    """Greedy Best-First Search to find a path."""
    def heuristic(word):
        return sum(1 for a, b in zip(word, end_word) if a != b)

    if start_word == end_word:
        return [start_word]

    open_list = []
    heapq.heappush(open_list, (heuristic(start_word), [start_word]))
    visited = set([start_word])

    while open_list:
        _, path = heapq.heappop(open_list)
        current_word = path[-1]

        if current_word == end_word:
            return path

        neighbors = [(heuristic(neighbor), neighbor) for neighbor in get_neighbors(current_word) if neighbor not in visited]
        neighbors.sort()  # Sort by heuristic value

        for _, neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                heapq.heappush(open_list, (heuristic(neighbor), path + [neighbor]))

    return []

def get_neighbors(word, mode="normal"):
    """Find all valid words differing by one letter."""
    neighbors = []
    for i in range(len(word)):
        for letter in "abcdefghijklmnopqrstuvwxyz":
            new_word = word[:i] + letter + word[i+1:]
            if new_word in valid_words and new_word != word:
                # In challenge mode, check if the word is banned
                if mode == "challenge" and new_word in banned_words:
                    continue
                neighbors.append(new_word)
    return neighbors

def generate_graph(path):
    """Generate a graph visualization of the word ladder path."""
    G = nx.Graph()
    
    # Add nodes (words)
    for word in path:
        G.add_node(word)
    
    # Add edges (connections between words)
    for i in range(len(path) - 1):
        G.add_edge(path[i], path[i+1])
    
    # Create the visualization
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, edge_color='gray', width=2, font_size=10)
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Convert to base64 for embedding in HTML
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{data}"

def calculate_score(moves, optimal_path_length, difficulty):
    """Calculate score based on moves, optimal path length, and difficulty."""
    if moves == optimal_path_length:
        # Perfect score if using optimal number of moves
        base_score = 1000
    else:
        # Decrease score as moves increase beyond optimal
        base_score = max(100, 1000 - (moves - optimal_path_length) * 50)
    
    # Multiply by difficulty factor
    difficulty_factor = 1.0  # easy
    if difficulty == "medium":
        difficulty_factor = 1.5
    elif difficulty == "hard":
        difficulty_factor = 2.0
    elif difficulty == "challenge":
        difficulty_factor = 3.0
    
    return int(base_score * difficulty_factor)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_game', methods=['POST'])
def start_game():
    data = request.json
    username = data.get("username")
    difficulty = data.get("difficulty")
    custom_words = data.get("custom_words", False)
    start_word_custom = data.get("start_word")
    end_word_custom = data.get("end_word")

    if not username:
        return jsonify({"error": "Username is required."})
    
    # Set difficulty parameters
    if difficulty == "easy":
        words_length = [3, 4]
        max_moves = None
        mode = "normal"
    elif difficulty == "medium":
        words_length = [5, 6]
        max_moves = random.randint(10, 12)
        mode = "normal"
    elif difficulty == "hard":
        words_length = [6, 8]  # Adjusted to be more challenging
        max_moves = random.randint(10, 15)  # Increased max moves for longer words
        mode = "normal"
    elif difficulty == "challenge":
        words_length = [4, 7]
        max_moves = random.randint(6, 10)
        mode = "challenge"
    else:
        return jsonify({"error": "Invalid difficulty level"}), 400

    # Handle custom words input
    if custom_words and start_word_custom and end_word_custom:
        start_word = start_word_custom.lower()
        end_word = end_word_custom.lower()
        
        # Validate the custom words
        if start_word not in valid_words:
            return jsonify({"error": f"'{start_word}' is not in our dictionary."})
        if end_word not in valid_words:
            return jsonify({"error": f"'{end_word}' is not in our dictionary."})
        if len(start_word) != len(end_word):
            return jsonify({"error": "Start word and end word must be the same length."})
        if start_word == end_word:
            return jsonify({"error": "Start word and end word must be different."})
        
        # Check if a path exists
        path = bfs(start_word, end_word)
        if not path:
            return jsonify({"error": "No valid path exists between these words."})
            
        # Set max_moves based on path length for custom games
        optimal_path_length = len(path) - 1
        max_moves = optimal_path_length + 5  # Give 5 extra moves beyond optimal
    else:
        # Generate random words based on difficulty
        attempts = 0
        min_path_length = 3  # Minimum path length for generated words
        max_path_length = 12 if difficulty == "hard" else 8  # Longer paths for hard mode
        
        while attempts < 50:
            length = random.choice(words_length)
            if length not in words_by_length or len(words_by_length[length]) < 2:
                continue
            
            available_words = words_by_length[length]
            start_word = random.choice(available_words)
            end_word = random.choice(available_words)
            
            if start_word == end_word:
                continue
                
            if mode == "challenge" and (start_word in banned_words or end_word in banned_words):
                continue
                
            path = bfs(start_word, end_word)
            path_length = len(path) if path else 0
            
            # Ensure path meets difficulty requirements
            if path and min_path_length <= path_length <= max_path_length:
                if difficulty == "hard" and path_length < 6:  # Ensure hard mode has longer paths
                    continue
                break
                
            attempts += 1
        
        if attempts >= 50:
            return jsonify({"error": "Could not generate a suitable word pair. Please try again."})

    # Calculate the optimal path length
    optimal_path_length = len(path) - 1  # Subtract 1 because we count moves, not words
    
    # Generate a game ID
    game_id = f"{username}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    return jsonify({
        "username": username,
        "start_word": start_word,
        "end_word": end_word,
        "path_length": optimal_path_length,
        "max_moves": max_moves,
        "mode": mode,
        "game_id": game_id
    })

@app.route('/check_word', methods=['POST'])
def check_word():
    """Check if a word is valid and differs by one letter."""
    data = request.json
    current_word = data.get("current_word")
    new_word = data.get("new_word")
    mode = data.get("mode", "normal")
    
    # Check if the word is banned in challenge mode
    if mode == "challenge" and new_word in banned_words:
        return jsonify({"valid": False, "message": "This word is banned in Challenge Mode!"})
    
    if new_word in valid_words and new_word in get_neighbors(current_word, mode):
        return jsonify({"valid": True})
    return jsonify({"valid": False, "message": "Invalid move. The word must differ by exactly one letter and be a valid dictionary word."})

@app.route('/hint', methods=['POST'])
def hint():
    """Provide a hint using a selected algorithm."""
    data = request.json
    current_word = data.get("current_word")
    end_word = data.get("end_word")
    algorithm = data.get("algorithm")
    mode = data.get("mode", "normal")

    if algorithm == "BFS":
        path = bfs(current_word, end_word)
    elif algorithm == "UCS":
        path = ucs(current_word, end_word)
    elif algorithm == "A*":
        path = a_star(current_word, end_word)
    elif algorithm == "Greedy":
        path = greedy_bfs(current_word, end_word)
    else:
        return jsonify({"error": "Invalid algorithm selection."})

    if path and len(path) > 1:
        # Generate graph visualization of the path
        graph_image = generate_graph(path)
        return jsonify({"hint": path[1], "path": path, "graph": graph_image})
    return jsonify({"hint": None, "message": "No valid hint available."})

@app.route('/end_game', methods=['POST'])
def end_game():
    """End the game and calculate final score."""
    data = request.json
    moves = data.get("moves")
    optimal_path_length = data.get("optimal_path_length")
    difficulty = data.get("difficulty")
    game_id = data.get("game_id")
    
    score = calculate_score(moves, optimal_path_length, difficulty)
    
    # In a real application, you might want to save scores to a database
    # For now, we'll just return the calculated score
    return jsonify({
        "score": score,
        "optimal_path_length": optimal_path_length,
        "moves_taken": moves,
        "difficulty": difficulty
    })

@app.route('/custom_path', methods=['POST'])
def custom_path():
    """Check if a path exists between two custom words."""
    data = request.json
    start_word = data.get("start_word").lower()
    end_word = data.get("end_word").lower()
    
    if start_word not in valid_words:
        return jsonify({"valid": False, "message": f"'{start_word}' is not in our dictionary."})
    if end_word not in valid_words:
        return jsonify({"valid": False, "message": f"'{end_word}' is not in our dictionary."})
    
    path = bfs(start_word, end_word)
    if path:
        return jsonify({"valid": True, "path_length": len(path) - 1})
    else:
        return jsonify({"valid": False, "message": "No valid path exists between these words."})

@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    """Get the leaderboard data - in a real app, this would fetch from a database."""
    # This is a placeholder - in a real application, you'd fetch from a database
    mock_leaderboard = [
        {"username": "player1", "score": 2500, "difficulty": "hard"},
        {"username": "player2", "score": 1800, "difficulty": "medium"},
        {"username": "player3", "score": 1200, "difficulty": "easy"},
        {"username": "player4", "score": 3100, "difficulty": "challenge"},
    ]
    return jsonify({"leaderboard": mock_leaderboard})

if __name__ == "__main__":
    app.run(debug=True)