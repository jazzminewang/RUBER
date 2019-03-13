
import subprocess32 as subprocess

gru_num_units=[64, 128, 256, 512]
init_learning_rate=[0.001, 0.0015, 0.01, 0.015, 0.1, 0.15]
margins=[0.1, 0.25, 0.5, 0.75]

for gru_num_unit in gru_num_units:
    for init_learning in init_learning_rate:
        for margin in margins:
            name_learning = init_learning * 1000
            name_margin = margin * 100

            command = "tmux new-session -d -s personachat_gru_" + str(gru_num_unit) + "_learning_" + str(int(name_learning)) + "_margin_" + str(int(name_margin))+ "_batchnorm" 
            command = command.split(" ")
	    command.append("python hybrid_evaluation.py personachat personachat train -gru_num_units=gru_num_unit -init_learning_rate=init_learning -margin=margin -batchnorm=True")

	    print(command)
            subprocess.run(command)

            command = "tmux new-session -d -s twitter_gru_" + str(gru_num_unit) + "_learning_" + str(int(name_learning)) + "_margin_" + str(int(name_margin)) + "_batchnorm"
            command = command.split(" ")
	    command.append("python hybrid_evaluation.py twitter twitter train -gru_num_units=gru_num_unit -init_learning_rate=init_learning -margin=margin -batchnorm=True")
            subprocess.run(command)

            command = "tmux new-session -d -s personachat_gru_" + str(gru_num_unit) + "_learning_" + str(int(name_learning)) + "_margin_" + str(int(name_margin)) 
            command = command.split(" ")
	    command.append("python hybrid_evaluation.py personachat personachat train -gru_num_units=gru_num_unit -init_learning_rate=init_learning -margin=margin")
            subprocess.run(command)

            command = "tmux new-session -d -s twitter_gru_" + str(gru_num_unit) + "_learning_" + str(int(name_learning)) + "_margin_" + str(int(name_margin))
	    command = command.split(" ")
	    command.append("python hybrid_evaluation.py twitter twitter train -gru_num_units=gru_num_unit -init_learning_rate=init_learning -margin=margin")
            subprocess.run(command)
