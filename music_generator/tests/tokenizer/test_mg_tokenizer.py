from music_generator.tokenizing.tokenizer.MGTokenizer import MGTokenizer


load_path = "./music_generator/src/tokenizer/mgt_tokenizer_v1.model"

text = """<SOS>X:6
L:1/8
M:3/4
K:Emin
E/F/G/A/ B2 e2 | ^d/e/f/d/ B2 B2 | cB AG FE | E/F/^G/A/ B2 B2 | gb fa eg | fe ^df BA | gb fa eg | 
fe/^d/ e2 E2 :: dd/e/ d2 B2 | cB/A/ B2 G2 | dd/e/ d2 B2 | cB/A/ B4 | cB AG FE | E/F/G/A/ B4 | 
gb fa eg | fe ^df BA | gb fa eg | fe/^d/ e4 :|<EOS>"""


tokenizer = MGTokenizer()
tokenizer.load(load_path)

encoded_tokens = tokenizer.encode("<SOS> E2 EF E2 B2 |1 efe^d e2 e2 <EOS>")
print("Encoded Tokens:", encoded_tokens)

decoded_text = tokenizer.decode(encoded_tokens)
print("Decoded Text:", decoded_text)

vocab_size = len(tokenizer.vocab)
print("Vocabulary Size:", vocab_size)