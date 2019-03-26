from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='IWE')
    parser.add_argument('--data', type=str, default='./data/document_passages.json',
                        help='location of the data corpus')
    parser.add_argument('--doc_no', type=str, default='1', help="number for particular WPQA doc")
    parser.add_argument('--write_emb', type=str, default='./embeddings/emb_', help='path of writing the embedding')
    parser.add_argument('--write_words', type=str, default='./embeddings/words_', help='path of writing the words')
    parser.add_argument('--K', type=int, default=20, help='number of negative samples')
    parser.add_argument('--N', type=int, default=5, help='number of context size')
    parser.add_argument('--both_sides', type=bool, default=True, help='if context size is on both sides or one')
    parser.add_argument('--window_size', type=int, default=3 , help='window size of convlutional filter')
    parser.add_argument('--emb_dim', type=int, default=100 , help='Embedding size')
    parser.add_argument('--hidden', type=int, default=1000 , help='dimension of hidden layer of embedding')
    parser.add_argument('--lr', type=float, default=0.1 , help='Learning Rate of optimizer')
    parser.add_argument('--epoch', type=int, default=50 , help='Embedding size')
    parser.add_argument('--cuda', type=int, default=0 , help='Cuda device')
    parser.add_argument('--gamma', type=int, default=1 , help='Temperature Parameter for Softmax')
    



    args = parser.parse_args()

    return args
