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


    train_filename = './rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.train'
    valid_filename = './rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.dev'
    test_filename = './rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.test'

    if not args.toy:
        train_all_story = roc_data.get_roc_graph(train_filename)
        graphs_train = roc_data.process_graph(train_all_story)

        valid_all_story = roc_data.get_roc_graph(valid_filename)
        graphs_validate = roc_data.process_graph(valid_all_story)

        test_all_story = roc_data.get_roc_graph(test_filename)
        graphs_test = roc_data.process_graph(test_all_story)
    else:
        train_all_story = roc_data.get_roc_graph(train_filename)
        graphs_train = roc_data.process_graph(train_all_story[:100])

        valid_all_story = roc_data.get_roc_graph(valid_filename)
        graphs_validate = roc_data.process_graph(valid_all_story[:10])

        test_all_story = roc_data.get_roc_graph(test_filename)
        graphs_test = roc_data.process_graph(test_all_story[:10])

    #graphs_train = graphs
    all_story = train_all_story + valid_all_story + test_all_story

    #graph_validate_len = 0
    #for graph in graphs_validate:
    #    graph_validate_len += graph.number_of_nodes()
    #graph_validate_len /= len(graphs_validate)
    #print('graph_validate_len', int(graph_validate_len))

    #graph_test_len = 0
    #for graph in graphs_test:
    #    graph_test_len += graph.number_of_nodes()
    #graph_test_len /= len(graphs_test)
    #print('graph_test_len', int(graph_test_len))
    
    #for g in graphs:
    #    print(g.nodes())
    #    print(g.number_of_nodes())

    # filter those imcomplete graphs
    graphs_new_train, graphs_new_validate, graphs_new_test = [],[],[]
    for g in graphs_train:
        if g.number_of_nodes() == 10:
            graphs_new_train.append(g)
    for g in graphs_validate:
        if g.number_of_nodes() == 10:
            graphs_new_validate.append(g)
    for g in graphs_test:
        if g.number_of_nodes() == 10:
            graphs_new_test.append(g)

    graphs = graphs_new_train + graphs_new_validate + graphs_new_test


    args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    args.min_num_node = min([graphs[i].number_of_nodes() for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])
    
    graphs_train = graphs_new_train
    graphs_validate = graphs_new_validate
    graphs_test = graphs_new_test

    # args.max_num_node = 2000
    # show graphs statistics
    print('training set: {}, validation set: {}, testing set: {}'.format(len(graphs_train),len(graphs_validate),len(graphs_test)))
    print('max/min number node: {} / {}'.format(args.max_num_node, args.min_num_node))
    print('max/min number edge: {} / {}'.format(max_num_edge,min_num_edge))
    print('max previous node: {}'.format(args.max_prev_node))

    # save ground truth graphs
    ## To get train and test set, after loading you need to manually slice
    save_graph_list(graphs_train, args.graph_save_path + 'data/' + args.fname_train + '0.dat')
    save_graph_list(graphs_validate, args.graph_save_path + 'data/' + args.fname_valid + '0.dat')
    save_graph_list(graphs_test, args.graph_save_path + 'data/' + args.fname_test + '0.dat')
    print('train graphs saved at: ', args.graph_save_path + 'data/' + args.fname_train + '0.dat')
    print('validation graphs saved at: ', args.graph_save_path + 'data/' + args.fname_valid + '0.dat')
    print('test graphs saved at: ', args.graph_save_path + 'data/' + args.fname_test + '0.dat')

    ### comment when normal training, for graph completion only
    # p = 0.5
    # for graph in graphs_train:
    #     for node in list(graph.nodes()):
    #         # print('node',node)
    #         if np.random.rand()>p:
    #             graph.remove_node(node)
        # for edge in list(graph.edges()):
        #     # print('edge',edge)
        #     if np.random.rand()>p:
        #         graph.remove_edge(edge[0],edge[1])

    ### dataset initialization
    if 'nobfs' in args.note:
        print('nobfs')
        dataset = Graph_sequence_sampler_pytorch_nobfs(graphs_train, max_num_node=args.max_num_node)
        args.max_prev_node = args.max_num_node-1
    if 'barabasi_noise' in args.graph_type:
        print('barabasi_noise')
        dataset = Graph_sequence_sampler_pytorch_canonical(graphs_train,max_prev_node=args.max_prev_node)
        args.max_prev_node = args.max_num_node - 1
    else:
        dataset = Graph_sequence_sampler_pytorch(graphs_train,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node)
        args.max_prev_node = dataset.max_prev_node
    
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                     num_samples=args.batch_size*args.batch_ratio, replacement=True)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                               sampler=sample_strategy)
    ### model initialization
    ## Graph RNN VAE model
    # lstm = LSTM_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_lstm,
    #                   hidden_size=args.hidden_size, num_layers=args.num_layers).cuda()

    if 'GraphRNN_VAE_conditional' in args.note:
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

    ### start training
    train(args, dataset_loader, rnn, output)

    ### graph completion
    # train_graph_completion(args,dataset_loader,rnn,output)

    ### nll evaluation
    # train_nll(args, dataset_loader, dataset_loader, rnn, output, max_iter = 200, graph_validate_len=graph_validate_len,graph_test_len=graph_test_len)

