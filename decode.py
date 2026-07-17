import tiktoken
import sys

def main():
    try:
        with open('generated.txt', 'r') as f:
            tokens_str = f.read().split()
        
        tokens = [int(x) for x in tokens_str if x.strip()]
        
        enc = tiktoken.get_encoding("gpt2")
        text = enc.decode(tokens)
        
        print("\n\n=== GENERATED TEXT ===\n")
        print(text)
        print("\n======================\n")
        
    except Exception as e:
        print(f"Error decoding: {e}")

if __name__ == '__main__':
    main()
