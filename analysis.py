# this file is used to plot images
from main import *

args = Args()
print(args.graph_type, args.note)
# epoch = 16000
epoch = args.load_epoch
sample_time = 1


# give file name and figure name
#fname_real = args.graph_save_path + args.fname_real + str(0)
fname_pred = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time)
figname = args.figure_save_path + args.fname + str(epoch) +'_'+str(sample_time)

#print(fname_real)
print(fname_pred)
print(figname)

# load data
train_graph_list = load_graph_list(fname_pred + '.dat')
#graph_pred_len_list_raw = np.array([len(graph_pred_list_raw[i]) for i in range(len(graph_pred_list_raw))])
#print(len(graph_pred_list_raw))
#print(len(graph_pred_len_list_raw))

print(len(train_graph_list))
group_num = len(train_graph_list) / 25
for num in range(int(group_num)):
    draw_graph_list(train_graph_list[num*25:(num+1)*25], row=5, col=5, fname=figname + "_" + str(num) + "_")



'''graph_pred_list = graph_pred_list_raw
graph_pred_len_list = graph_pred_len_list_raw


pred_order = np.argsort(graph_pred_len_list)[::-1]
graph_pred_list = [graph_pred_list[i] for i in pred_order]'''

#print('pred average nodes', sum([graph_pred_list[i].number_of_nodes() for i in range(len(graph_pred_list))])/len(graph_pred_list))
#print('num of pred graphs', len(graph_pred_list))

# draw all graphs
'''for iter in range(8):
    print('iter', iter)
    graph_list = []
    for i in range(8):
        index = 32 * iter + i
        graph_list.append(graph_pred_list[index])
        print('pred', graph_pred_list[index].number_of_nodes())

    draw_graph_list(graph_list, row=4, col=4, fname=figname + '_' + str(iter)+'_pred')'''

'''# draw all graphs
for iter in range(8):
    print('iter', iter)
    graph_list = []
    for i in range(8):
        index = 16 * iter + i
        graph_list.append(graph_real_list[index])
        print('real', graph_real_list[index].number_of_nodes())

    draw_graph_list(graph_list, row=4, col=4, fname=figname + '_' + str(iter)+'_real')'''


