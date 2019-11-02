from main import *
import networkx
args = Args()


train_graph_lname = args.graph_save_path + 'data/' + args.fname_train + '0.dat'
valid_graph_lname = args.graph_save_path + 'data/' + args.fname_valid + '0.dat'
test_graph_lname = args.graph_save_path + 'data/' + args.fname_test + '0.dat'

train_graph_fname = args.figure_save_path + 'data/' + args.fname_train
valid_graph_fname = args.figure_save_path + 'data/' + args.fname_valid
test_graph_fname = args.figure_save_path + 'data/' + args.fname_test

print("Training Graphs datalist filename: " + train_graph_lname)
print("Validation Graphs datalist filename: " + valid_graph_lname)
print("Testing Graphs datalist filename: " + test_graph_lname)

print("Training visualised graph save filename: " + train_graph_fname)
print("Validation visualised graph save filename: " + valid_graph_fname)
print("Testing visualised graph save filename: " + test_graph_fname)

train_graph_list = load_graph_list(train_graph_lname)
valid_graph_list = load_graph_list(valid_graph_lname)
test_graph_list = load_graph_list(test_graph_lname)

group_num = len(train_graph_list) / 25
for num in range(int(group_num)):
    draw_graph_list(train_graph_list[num*25:(num+1)*25], row=5, col=5, fname=train_graph_fname + "_" + str(num) + "_")

group_num = len(valid_graph_list) / 25
if group_num == 0:
    for num in range(int(group_num)):
        draw_graph_list(valid_graph_list, row=5, col=5, fname=valid_graph_fname + "_" + str(num) + "_")
else:
    draw_graph_list(valid_graph_list, row=5, col=5, fname=valid_graph_fname)

group_num = len(test_graph_list) / 25
if group_num == 0:
    for num in range(int(group_num)):
        draw_graph_list(test_graph_list, row=5, col=5, fname=test_graph_fname + "_" + str(num) + "_")
else:
    draw_graph_list(test_graph_list, row=5, col=5, fname=test_graph_fname)

