# this file is used to plot images
from main import *

args = Args()
print(args.graph_type, args.note)
# epoch = 16000
epoch = args.load_epoch
sample_time = 3


# give file name and figure name
fname_pred = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time)
figname = args.figure_save_path + args.fname + str(epoch) +'_'+str(sample_time)

print(fname_pred)
print(figname)

# load data
train_graph_list = load_graph_list(fname_pred + '.dat')
print(len(train_graph_list))
group_num = len(train_graph_list) / 25
for num in range(int(group_num)):
    draw_graph_list(train_graph_list[num*25:(num+1)*25], row=5, col=5, fname=figname + "_" + str(num) + "_")



