import tiktoken
import base64

def main():
    enc = tiktoken.get_encoding("gpt2")
    
    with open("vocab.txt", "w", encoding="utf-8") as f:
        # GPT-2 vocab size is 50257
        for i in range(enc.n_vocab):
            try:
                # We decode to bytes and then encode as base64 to safely store all raw byte combinations
                token_bytes = enc.decode_single_token_bytes(i)
                b64_str = base64.b64encode(token_bytes).decode('utf-8')
                f.write(f"{i} {b64_str}\n")
            except Exception as e:
                pass
                
if __name__ == '__main__':
    main()
