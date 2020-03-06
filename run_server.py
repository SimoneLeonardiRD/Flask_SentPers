from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer


print("prova")
args = get_args_parser().parse_args(['-model_dir','../dataset/multi_cased_L-12_H-768_A-12','-port', '5555', '-port_out', '5556','-max_seq_len','NONE', '-mask_cls_sep','-cpu','-num_worker=1', '-pooling_strategy', 'CLS_TOKEN'])
server = BertServer(args)
server.start()
