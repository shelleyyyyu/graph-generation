from train import *
import roc_data.data_process_keyword as roc_data
if __name__ == '__main__':
    # All necessary arguments are defined in args.py
    args = Args()
    # check if necessary directories exist
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.graph_save_path + "/data"):
        os.makedirs(args.graph_save_path + "/data")
    if not os.path.isdir(args.figure_save_path + "/data"):
        os.makedirs(args.figure_save_path + "/data")
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)
    if not os.path.isdir(args.nll_save_path):
        os.makedirs(args.nll_save_path)

    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    # logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)
    if args.clean_tensorboard:
        if os.path.isdir("tensorboard"):
            shutil.rmtree("tensorboard")
    configure("tensorboard/run"+time, flush_secs=5)


    test_filename = './rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.test'

    if not args.toy:
        test_all_story = roc_data.get_roc_graph(test_filename)
        graphs_test = roc_data.process_graph(test_all_story)
    else:
        test_all_story = roc_data.get_roc_graph(test_filename)
        graphs_test = roc_data.process_graph(test_all_story[:10])

    # filter those imcomplete graphs
    graphs_new_test = []
    for g in graphs_test:
        if g.number_of_nodes() == 10:
            graphs_new_test.append(g)

    graphs_test = graphs_new_test

    args.max_num_node = max([graphs_test[i].number_of_nodes() for i in range(len(graphs_test))])
    args.min_num_node = min([graphs_test[i].number_of_nodes() for i in range(len(graphs_test))])
    max_num_edge = max([graphs_test[i].number_of_edges() for i in range(len(graphs_test))])
    min_num_edge = min([graphs_test[i].number_of_edges() for i in range(len(graphs_test))])
    

    print('testing set: {}'.format(len(graphs_test)))
    print('max/min number node: {} / {}'.format(args.max_num_node, args.min_num_node))
    print('max/min number edge: {} / {}'.format(max_num_edge,min_num_edge))
    print('max previous node: {}'.format(args.max_prev_node))

    test_dataset = Graph_sequence_sampler_pytorch(graphs_test,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node)
    args.max_prev_node = test_dataset.max_prev_node
    
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(test_dataset) for i in range(len(test_dataset))],
                                                                     num_samples=args.test_batch_size*args.batch_ratio, replacement=True)
    dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                               sampler=sample_strategy)

    if 'GraphRNN_VAE' in args.note:
        if torch.cuda.is_available():
            rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                            hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                            has_output=False).cuda()
            output = MLP_VAE_conditional_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output, y_size=args.max_prev_node).cuda()
        else:
            rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                            hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                            has_output=False)
            output = MLP_VAE_conditional_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output, y_size=args.max_prev_node)
            
    elif 'GraphRNN_MLP' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=False).cuda()
        output = MLP_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output, y_size=args.max_prev_node).cuda()
    elif 'GraphRNN_RNN' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=True, output_size=args.hidden_size_rnn_output).cuda()
        output = GRU_plain(input_size=1, embedding_size=args.embedding_size_rnn_output,
                           hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
                           has_output=True, output_size=1).cuda()

    test_GraphRNN_VAE(args, rnn, output)


