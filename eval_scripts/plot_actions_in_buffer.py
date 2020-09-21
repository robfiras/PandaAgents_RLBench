import os
import sys
import sqlite3
import getopt

import numpy as np
import matplotlib.pyplot as plt


def help_mess():
    mess = "Please define an absolute path to the database!\n" \
           "Example: python plot_actions_in_buffer.py --log_dir /path/to/database.db --n_samples 500"
    return mess


abs_path_to_database = None
n_samples_to_plot = 1000     # we don't want to plot all samples in the buffer

# get the arguments
try:
    opts, args = getopt.getopt(sys.argv[1:], "hl:n:", ["help", "log_dir=", "n_samples="])
except getopt.GetoptError as e:
    print("\n Error: ", e, "\n")

for opt, arg in opts:
    if opt in ("-h", "--help"):
        print(help_mess())
        sys.exit()
    elif opt in ("-l", "--log_dir"):
        abs_path_to_database = arg
    elif opt in ("-n", "--n_samples"):
        n_samples_to_plot = int(arg)

# check if path is still None
if not abs_path_to_database:
    raise ValueError(help_mess())

# check if database exists
if not os.path.exists(abs_path_to_database):
    raise FileNotFoundError(("Could not find database at %s. Please double-check!" % abs_path_to_database))

# create connection to database
conn = sqlite3.connect(abs_path_to_database)
c = conn.cursor()

# get all action in buffer
input_str = "Select Action_0, Action_1, Action_2, Action_3, Action_4, Action_5, Action_6, Action_7 from data"
c.execute(input_str)
print("Fetching data ...")
data = np.array(c.fetchall())
print("Done!")

# plot data
print("Starting plotting ...")
idx = np.sort(np.random.randint(len(data), size=n_samples_to_plot))  # choose samples randomly from buffer
action_0 = data[idx, 0]
action_1 = data[idx, 1]
action_2 = data[idx, 2]
action_3 = data[idx, 3]
action_4 = data[idx, 4]
action_5 = data[idx, 5]
action_6 = data[idx, 6]
action_7 = data[idx, 7]

fig, axs = plt.subplots(2, 4)
axs[0, 0].plot(idx, action_0)
axs[0, 0].set_title('Action_0')
axs[0, 1].plot(idx, action_1)
axs[0, 1].set_title('Action_1')
axs[0, 2].plot(idx, action_2)
axs[0, 2].set_title('Action_2')
axs[0, 3].plot(idx, action_3)
axs[0, 3].set_title('Action_3')
axs[1, 0].plot(idx, action_4)
axs[1, 0].set_title('Action_4')
axs[1, 1].plot(idx, action_5)
axs[1, 1].set_title('Action_5')
axs[1, 2].plot(idx, action_6)
axs[1, 2].set_title('Action_6')
axs[1, 3].plot(idx, action_7)
axs[1, 3].set_title('Action_7')

plt.show()
