import os

# data loading
def read_csv(path="train.csv"):
    data = []
    
    # Try various likely paths
    possible_paths = [
        path,
        os.path.join(os.path.dirname(__file__), "data set", "train.csv"),
        os.path.join(os.path.dirname(__file__), "train.csv"),
        "data set/train.csv",
        "/Users/Noura/Desktop/work/college/level 002/dump/level 2/second sem/ML/Titanic project/data set/train.csv",
        "/Users/Noura/Desktop/work/college/level 002/second sem/ML/Titanic project/data set/train.csv"
    ]
    
    chosen_path = None
    for p in possible_paths:
        if p and os.path.exists(p):
            chosen_path = p
            break
            
    if not chosen_path:
        # Fallback to the original path even if not found immediately
        chosen_path = "/Users/Noura/Desktop/work/college/level 002/second sem/ML/Titanic project/data set/train.csv"
        
    with open(chosen_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        for line in f:
            row = []
            value = ''
            inside_quotes = False
            for c in line:
                if c == '"':
                    inside_quotes = not inside_quotes
                elif c == ',' and not inside_quotes:
                    row.append(value)
                    value = ''
                else:
                    value += c
            row.append(value.strip())
            data.append(row)
    return header, data