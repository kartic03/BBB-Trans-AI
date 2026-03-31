import csv
from pathlib import Path

# Paths
pos_path = Path("../data/bbb_pos_dataset1.fasta")
neg_path = Path("../data/bbb_neg_dataset1.fasta")

out_csv = Path("../data/bbb_dataset.csv")
out_fasta = Path("../data/bbb_dataset.fasta")

def read_fasta(fp):
    sequences = []
    with open(fp, "r") as f:
        seq_id = None
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id is not None:
                    sequences.append((seq_id, "".join(seq)))
                seq_id = line[1:]
                seq = []
            else:
                seq.append(line)
        if seq_id is not None:
            sequences.append((seq_id, "".join(seq)))
    return sequences

pos_seqs = read_fasta(pos_path)
neg_seqs = read_fasta(neg_path)

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "Sequence", "BBB_label"])
    
    for sid, s in pos_seqs:
        writer.writerow([sid, s, 1])
    
    for sid, s in neg_seqs:
        writer.writerow([sid, s, 0])

with open(out_fasta, "w") as f:
    for sid, s in pos_seqs:
        f.write(f">{sid}\n{s}\n")
    for sid, s in neg_seqs:
        f.write(f">{sid}\n{s}\n")

print("✔ Dataset built successfully!")
print("Created:")
print(f"  - {out_csv}")
print(f"  - {out_fasta}")
