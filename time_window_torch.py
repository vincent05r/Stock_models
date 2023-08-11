def pandas_window(data, window_length=5):

    output = []

    for i in range(len(data) - window_length):
        output.append(data[i:i+window_length])
    
    return output
