import random
from flask import Flask, render_template, request, jsonify
import csv
from collections import deque
import heapq

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
    """A* Search Algorithm to find the shortest path."""
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

def get_neighbors(word):
    """Find all valid words differing by one letter."""
    neighbors = []
    for i in range(len(word)):
        for letter in "abcdefghijklmnopqrstuvwxyz":
            new_word = word[:i] + letter + word[i+1:]
            if new_word in valid_words and new_word != word:
                neighbors.append(new_word)
    return neighbors

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_game', methods=['POST'])
def start_game():
    data = request.json
    username = data.get("username")
    difficulty = data.get("difficulty")

    if not username:
        return jsonify({"error": "Username is required."})
    
    # Set difficulty parameters
    if difficulty == "easy":
        words_of_length = (3, 4)
        max_moves = None
    elif difficulty == "medium":
        words_of_length = (5, 6)
        max_moves = random.randint(10, 12)  # Example: 10-12 moves
    elif difficulty == "hard":
        words_of_length = (7, 10)
        max_moves = random.randint(8, 12)  # Example: 8-12 moves
    else:
        return jsonify({"error": "Invalid difficulty level"}), 400

    if len(words_of_length) < 2:
        return jsonify({"error": "Not enough words of the requested length."})
    
    # # Ensure start_word is not a banned word in Hard mode
    # if difficulty == "hard":
    #     while start_word in banned_words or end_word in banned_words:
    #         start_word = random.choice(words_of_length)
    #         end_word = random.choice(words_of_length)

    while True:
        start_word = random.choice(words_of_length)
        end_word = random.choice(words_of_length)
        if start_word != end_word:
            path = bfs(start_word, end_word)
            if path:
                break

    return jsonify({
        "username": username,
        "start_word": start_word,
        "end_word": end_word,
        "path_length": len(path)
    })

@app.route('/check_word', methods=['POST'])
def check_word():
    """Check if a word is valid and differs by one letter."""
    data = request.json
    current_word = data.get("current_word")
    new_word = data.get("new_word")

    if new_word in valid_words and new_word in get_neighbors(current_word):
        return jsonify({"valid": True})
    return jsonify({"valid": False})

@app.route('/hint', methods=['POST'])
def hint():
    """Provide a hint using a selected algorithm."""
    data = request.json
    current_word = data.get("current_word")
    end_word = data.get("end_word")
    algorithm = data.get("algorithm")

    if algorithm == "BFS":
        path = bfs(current_word, end_word)
    elif algorithm == "UCS":
        path = ucs(current_word, end_word)
    elif algorithm == "A*":
        path = a_star(current_word, end_word)
    else:
        return jsonify({"error": "Invalid algorithm selection."})

    if path and len(path) > 1:
        return jsonify({"hint": path[1]})
    return jsonify({"hint": None, "message": "No valid hint available."})

if __name__ == "__main__":
    app.run(debug=True)
