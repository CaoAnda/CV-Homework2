import argparse  
  
def get_args(parser=argparse.ArgumentParser()):
    parser.add_argument('--device', type=str, default='cuda:0')  
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--decoding_method', type=str, default='deconv', choices=['upsample', 'deconv'])
    parser.add_argument('--num_classes', type=int, default=2, choices=[2, 11])
    parser.add_argument('--output', action='store_true', default=True)  
  
    opt = parser.parse_args()  
    if opt.output:  
        print(opt)
    return opt  
  
if __name__ == '__main__':  
    opt = get_args()