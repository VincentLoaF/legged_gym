import matplotlib.pyplot as plt
import numpy as np

# # Create x-axis values
# x = range(501)
#
# # Create y-axis values based on curriculum factor
# y = []
# for i in x:
#     if i < 100:
#         y.append(0)
#     elif i < 300:
#         y.append((i - 100) / 200)
#     else:
#         y.append(1)
#
# # Create and show plot
# plt.plot(x, y)
# plt.xlabel('Training Iteration')
# plt.ylabel('Curriculum Factor')
# plt.title('Reward Curriculum Factor over Training')
# plt.savefig('reward_curriculum.png', dpi=200)

# # Create x-axis values
# x = range(501)
#
# # Create y-axis values based on curriculum factor
# y = []
# for i in x:
#     if i < 100:
#         y.append(0)
#     elif i < 500:
#         y.append((i - 100) / 400)
#     else:
#         y.append(1)
#
# # Create and show plot
# plt.plot(x, y)
# plt.xlabel('Training Iteration')
# plt.ylabel('Curriculum Factor')
# plt.title('Noise Curriculum Factor over Training')
# plt.savefig('noise_curriculum.png', dpi=200)

# # Define the function
# def f(x, sigma):
#     return np.exp(-x**2 / sigma)
#
# # Find x value (deviation) for given y (reward) and sigma
# def find_x(y, sigma):
#     return np.sqrt(-sigma * np.log(y))
#
# # Define the range of x values
# x = np.linspace(-2, 2, 1000)
#
# # Define the 80% y-value
# y_80 = 0.8
#
# # Calculate the corresponding x values (deviation) for each sigma value at the 80% y-value
# x_sigma_1 = find_x(y_80, 1)
# x_sigma_05 = find_x(y_80, 0.5)
# x_sigma_025 = find_x(y_80, 0.25)
#
# print(x_sigma_1, x_sigma_05, x_sigma_025)
#
# # Plot the function
# plt.plot(x, f(x, 1), label='sigma = 1')
# plt.plot(x, f(x, 0.5), label='sigma = 0.5')
# plt.plot(x, f(x, 0.25), label='sigma = 0.25')
#
# # Add the horizontal dotted line at 80% y-value
# plt.axhline(y=y_80, linestyle='--', color='gray')
#
# # Add vertical dotted lines for each sigma value, where the velocity tracking is reaching 80%
# plt.axvline(x=x_sigma_1, linestyle='--', color='blue')
# plt.axvline(x=-x_sigma_1, linestyle='--', color='blue')
#
# plt.axvline(x=x_sigma_05, linestyle='--', color='orange')
# plt.axvline(x=-x_sigma_05, linestyle='--', color='orange')
#
# plt.axvline(x=x_sigma_025, linestyle='--', color='green')
# plt.axvline(x=-x_sigma_025, linestyle='--', color='green')
#
# plt.legend()
#
# plt.xlabel('Velocity Tracking Error')
# plt.ylabel('Velocity Reward')
# plt.title('Velocity Reward for Different Sigma Values')
# plt.savefig('tracking_sigma.png', dpi=200)

# # Data
# categories = ['Linear Velocity', 'Angular Velocity']
# with_penalty = [7.497, 3.539]
# without_penalty = [6.529, 3.029]
#
# # Bar plot configuration
# bar_width = 0.3
# x = np.arange(len(categories))
#
# # Plotting the bars
# fig, ax = plt.subplots()
# bar1 = ax.bar(x - bar_width / 2, with_penalty, bar_width, label='With Penalty Terms')
# bar2 = ax.bar(x + bar_width / 2, without_penalty, bar_width, label='Without Penalty Terms')
#
# # Customizing the plot
# ax.set_xticks(x)
# ax.set_xticklabels(categories)
# ax.set_ylim([0, 8])
# ax.set_ylabel('Command Velocity Tracking Reward')
# ax.set_title('Command Velocity Tracking Reward With and Without Penalty Terms')
# ax.legend()
#
# # Adding values on top of the bars
# def add_values_on_bars(bars):
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate('{:.3f}'.format(height),
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
# add_values_on_bars(bar1)
# add_values_on_bars(bar2)
#
# # Saving the plot as an image file
# plt.savefig('penalty_terms_comparison.png', dpi=200)

# # Data
# weights = [0.0, 0.1, 0.2, 0.5]
# linear_velocity_rewards = [7.497, 7.487, 7.404, 3.418]
# angular_velocity_rewards = [3.539, 3.579, 3.417, 1.629]
# power_loss = [20.8, 14.7, 13.6, 4.6]
#
# # Function to add value labels
# def add_labels1(x, y, labels):
#     for i, label in enumerate(labels):
#         plt.text(x[i], y[i]+0.3, f"{label:.1f}", ha="center", va="bottom", fontsize=9)
# def add_labels3(x, y, labels):
#     for i, label in enumerate(labels):
#         plt.text(x[i], y[i]+0.3, f"{label:.3f}", ha="center", va="bottom", fontsize=9)
#
# # Plotting the lines
# plt.plot(weights, linear_velocity_rewards, marker='o', label='Linear Velocity Reward')
# plt.plot(weights, angular_velocity_rewards, marker='o', label='Angular Velocity Reward')
# plt.plot(weights, power_loss, marker='o', label='Power Loss')
#
# # Adding value labels
# add_labels3(weights, linear_velocity_rewards, linear_velocity_rewards)
# add_labels3(weights, angular_velocity_rewards, angular_velocity_rewards)
# add_labels1(weights, power_loss, power_loss)
#
# # Customizing the plot
# plt.xlabel('Power Loss Penalty Term Weight')
# plt.ylabel('Reward and Power Loss')
# plt.title('Reward and Power Loss vs. Penalty Term Weight')
# plt.legend()
#
# # Saving the plot as an image file
# plt.savefig('power_penalty.png', dpi=200)
#
# # Displaying the plot
# plt.show()

# number_of_robots = [128, 256, 512, 1024, 2048, 4096]
# number_of_robots_str = ["128", "256", "512", "1024", "2048", "4096"]
# iteration_time_batch_49152 = [5.91, 3.42, 1.94, 1.09, 0.64, 0.46]
# iteration_time_batch_98304 = [11.82, 6.79, 3.94, 2.15, 1.18, 0.89]
#
# x = np.arange(len(number_of_robots))
#
# plt.plot(x, iteration_time_batch_49152, marker="o", label="Batch Size = 49152")
# plt.plot(x, iteration_time_batch_98304, marker="o", label="Batch Size = 98304")
#
# for i in range(len(number_of_robots)):
#     plt.text(x[i]-0.15, iteration_time_batch_49152[i]+0.15, str(iteration_time_batch_49152[i]))
#     plt.text(x[i]-0.15, iteration_time_batch_98304[i]+0.15, str(iteration_time_batch_98304[i]))
#
# plt.xlabel("Number of Robots")
# plt.ylabel("Iteration Time (s)")
# plt.title("Iteration Time vs Number of Robots")
# plt.xticks(x, number_of_robots_str)
# plt.legend()
# plt.savefig('iteration_time.png', dpi=200)
# plt.show()

number_of_robots = [128, 256, 512, 1024, 2048, 4096]
number_of_robots_str = ["128", "256", "512", "1024", "2048", "4096"]
memory = [3215, 3246, 3277, 3318, 3400, 3564]
gpu_memory = [3689, 3691, 3773, 3889, 4187, 4639]

x = np.arange(len(number_of_robots))

plt.plot(x, memory, marker="o", label="Main Memory")
plt.plot(x, gpu_memory, marker="o", label="GPU Memory")

for i in range(len(number_of_robots)):
    plt.text(x[i] - 0.15, memory[i] + 15, str(memory[i]))
    plt.text(x[i] - 0.15, gpu_memory[i] + 15, str(gpu_memory[i]))

plt.xlabel("Number of Robots")
plt.ylabel("Memory Usage (MB)")
plt.title("Memory Usage vs Number of Robots")
plt.xticks(x, number_of_robots_str)
plt.legend()
plt.savefig('memory_usage.png', dpi=200)
plt.show()




