import argparse  
  
def get_args(parser=argparse.ArgumentParser()):
    parser.add_argument('--device', type=str, default='cuda:0')  
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=3407)
    # parser.add_argument('--input_size', type=int, default=32)
    parser.add_argument('--output', action='store_true', default=True)  
  
    opt = parser.parse_args()  
    if opt.output:  
        print(opt)
    return opt  
  
if __name__ == '__main__':  
    opt = get_args()