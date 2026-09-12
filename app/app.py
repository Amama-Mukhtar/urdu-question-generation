import torch
import gradio as gr
import sentencepiece as spm
import torch.nn as nn

PAD, UNK, BOS, EOS = 0, 1, 2, 3
MAX_SRC = 60
device = "cpu"

class Encoder(nn.Module):
    def __init__(self, vocab, emb=256, hid=512, n_layers=2, drop=0.3):
        super().__init__()
        self.hid = hid
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.rnn = nn.LSTM(emb, hid, n_layers, batch_first=True, bidirectional=True, dropout=drop)
        self.fc_h = nn.Linear(hid * 2, hid)
        self.fc_c = nn.Linear(hid * 2, hid)
        self.drop = nn.Dropout(drop)

    def forward(self, x, lens):
        e = self.drop(self.embed(x))
        packed = nn.utils.rnn.pack_padded_sequence(e, lens.cpu(), batch_first=True)
        out, (h, c) = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        h = h.view(self.n_layers, 2, h.size(1), self.hid)
        h = torch.tanh(self.fc_h(torch.cat((h[:, 0], h[:, 1]), dim=2)))
        c = c.view(self.n_layers, 2, c.size(1), self.hid)
        c = torch.tanh(self.fc_c(torch.cat((c[:, 0], c[:, 1]), dim=2)))
        return out, (h, c)

class Attention(nn.Module):
    def __init__(self, enc_dim=1024, hid=512):
        super().__init__()
        self.w1 = nn.Linear(enc_dim, hid, bias=False)
        self.w2 = nn.Linear(hid, hid, bias=False)
        self.v = nn.Linear(hid, 1, bias=False)

    def forward(self, dec_h, enc_out, mask):
        score = self.v(torch.tanh(self.w2(dec_h).unsqueeze(1) + self.w1(enc_out))).squeeze(2)
        score = score.masked_fill(mask, float("-inf"))
        a = torch.softmax(score, dim=1)
        ctx = torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)
        return ctx, a

class Decoder(nn.Module):
    def __init__(self, vocab, emb=256, hid=512, n_layers=2, drop=0.3, enc_dim=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.attn = Attention(enc_dim, hid)
        self.rnn = nn.LSTM(emb + enc_dim, hid, n_layers, batch_first=True, dropout=drop)
        self.fc_out = nn.Linear(hid + enc_dim, vocab)
        self.drop = nn.Dropout(drop)

    def forward(self, tok, hid, enc_out, mask):
        h, c = hid
        e = self.drop(self.embed(tok)).unsqueeze(1)
        ctx, a = self.attn(h[-1], enc_out, mask)
        out, (h, c) = self.rnn(torch.cat((e, ctx.unsqueeze(1)), dim=2), (h, c))
        pred = self.fc_out(torch.cat((out.squeeze(1), ctx), dim=1))
        return pred, (h, c), a

class Seq2Seq(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

sp = spm.SentencePieceProcessor(model_file="ur_sp.model")
vocab = sp.get_piece_size()
model = Seq2Seq(Encoder(vocab), Decoder(vocab))
model.load_state_dict(torch.load("best.pt", map_location=device))
model.eval()


def mark_answer(sentence, answer):
    if answer.strip() and answer in sentence:
        return sentence.replace(answer, " <ans> " + answer + " </ans> ", 1)
    return sentence


def run_model(sentence, answer):
    text = " ".join(mark_answer(sentence, answer).split())
    ids = sp.encode(text)[:MAX_SRC]
    x = torch.tensor([ids])
    x_len = torch.tensor([len(ids)])

    with torch.no_grad():
        enc_out, hid = model.enc(x, x_len)
        mask = (x == PAD)

        tok = torch.tensor([BOS])
        greedy_ids = []
        for _ in range(30):
            pred, hid, _ = model.dec(tok, hid, enc_out, mask)
            tok = pred.argmax(1)
            if tok.item() == EOS:
                break
            greedy_ids.append(tok.item())
        greedy_out = sp.decode(greedy_ids)

        enc_out, hid = model.enc(x, x_len)
        beams = [([BOS], 0.0, hid)]
        done = []
        for _ in range(30):
            pool = []
            for seq, score, h in beams:
                tok = torch.tensor([seq[-1]])
                pred, h2, _ = model.dec(tok, h, enc_out, mask)
                logp = torch.log_softmax(pred, dim=1).squeeze(0)
                top_p, top_i = logp.topk(4)
                for p, i in zip(top_p.tolist(), top_i.tolist()):
                    if i == EOS:
                        done.append((seq + [i], score + p))
                    else:
                        pool.append((seq + [i], score + p, h2))
            pool.sort(key=lambda b: b[1] / len(b[0]), reverse=True)
            beams = pool[:4]
            if not beams:
                break
        done += [(s, sc) for s, sc, _ in beams]
        done.sort(key=lambda b: b[1] / len(b[0]), reverse=True)
        beam_ids = [t for t in done[0][0] if t not in (BOS, EOS)]
        beam_out = sp.decode(beam_ids)

    return greedy_out, beam_out


demo = gr.Interface(
    fn=run_model,
    inputs=[gr.Textbox(label="Urdu sentence"), gr.Textbox(label="answer (exact text from sentence)")],
    outputs=[gr.Textbox(label="greedy question"), gr.Textbox(label="beam question")],
    title="Urdu Question Generation",
)

if __name__ == "__main__":
    demo.launch()
