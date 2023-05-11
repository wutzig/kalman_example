import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import random
def init_bar():
    truth_plot = plt.bar(x_array + 0.1, the_truth, color = color_truth)
    # truth_plot = plt.bar(x_array + 0.1, the_truth, color = color_obs)
    bar_plot = plt.bar(x_array, model_data, color = color_model)
    plt.gca().locator_params(axis = 'x', nbins = 9)
    plt.gca().tick_params(axis = 'both', labelsize = 16)
    plt.title(f'Time step 0', fontsize = 26)
    plt.ylim((0, 3.2))
    plt.xticks(x_array)
    plt.grid(True)
    return truth_plot, bar_plot

def model_matrix_central(c = 1):
    model_matrix = np.eye(model_size, model_size)
    for j in range(model_size):
        model_matrix[j,(j+1) % model_size] = -c /2
        model_matrix[j,(j-1) % model_size] = c/2
    # print(model_matrix)
    return np.linalg.inv(model_matrix)

def model_matrix_euler():
    model_matrix = np.zeros((model_size, model_size))
    for j in range(model_size):
        model_matrix[j,(j+1) % model_size] = -1
    return model_matrix

def simple_truth():
    the_truth = np.zeros(x_array.shape)
    the_truth[central_idx]   = 2.5
    return the_truth

def hard_truth():
    the_truth = 0.5 * np.cos(x_array * 2 * np.pi / model_size) + 2.5
    return the_truth

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    # plt.style.use('dark_background')
    color_truth = 'lightskyblue'
    color_obs   = 'blue'
    color_model = 'lightcoral'
    color_focus = 'red'

    velocity     = 1
    model_size   = 99
    x_array      = np.linspace(1, model_size, num = model_size, endpoint = True)
    
    obs_var      = 0.5
    model_var    = 0.5

    model_data   = 0.5 * np.sin(x_array * 2 * np.pi / model_size) + 1.5
    model_cov    = model_var * np.eye(model_size, model_size)
    model_matrix = model_matrix_central()
    reanalysis_cov = np.zeros((model_size, model_size))

    fig         = plt.figure(figsize=(19.2, 10.8), dpi = 150)
    central_idx = 4

    # the_truth   = simple_truth()
    the_truth   = hard_truth()

    truth_plot, bar_plot = init_bar()

    # fig.savefig('initial.png')
    bar_plot[central_idx].set_color(color_focus)
    # fig.savefig('initial_reanalysis.png')
    def update(frame):
        global model_matrix, model_data, the_truth, reanalysis_cov, central_idx
        
        # Advance the model
        # model_data = np.dot(model_matrix, model_data)
        model_data = np.roll(model_data, -1)
        
        # Update the truth
        the_truth = np.roll(the_truth, -1)

        # Create random observations from the truth
        obs_size     = random.randint(0, 10)
        observations = np.zeros((obs_size,))
        obs_model    = np.zeros((obs_size, model_size))

        for bar, val in zip(truth_plot, the_truth):
            bar.set_height(val)
            bar.set_color(color_truth)
        for ob in range(obs_size):
            idx = random.randint(0, model_size - 1)
            # idx = 4
            observations[ob]   = the_truth[idx]
            obs_model[ob, idx] = 1
            truth_plot[idx].set_color(color_obs)

        # The observation covariance
        obs_cov = obs_var * np.eye(obs_size, obs_size)

        # Predicted covariance
        model_matrix = model_matrix_central()
        predict_cov = np.dot(model_matrix, reanalysis_cov)
        predict_cov = np.dot(predict_cov, model_matrix.T)
        predict_cov += model_cov

        # Innovation matrix
        innov_mat   = np.dot(np.dot(obs_model, predict_cov), obs_model.T)
        innov_mat   += obs_cov

        # Kalman matrix
        kalman_gain = np.dot(predict_cov, obs_model.T)
        kalman_gain = np.dot(kalman_gain, np.linalg.inv(innov_mat))

        # Reanalysis
        model_data += np.dot(kalman_gain, observations - np.dot(obs_model, model_data))
        reanalysis_cov = np.dot(np.eye(model_size, model_size) - np.dot(kalman_gain, obs_model), predict_cov)
        print(reanalysis_cov, end = f"\n{12*'*'}\n")
        plt.title(f'Time step {frame+1}', fontsize = 26)
        for bar, val in zip(bar_plot, model_data):
            bar.set_height(val)
        bar_plot[int(central_idx)].set_color(color_model)
        central_idx = (central_idx - velocity) % model_size
        bar_plot[int(central_idx)].set_color(color_focus)
    
    animation = anim.FuncAnimation(fig, update, frames = model_size * 4, repeat = False, interval = 100)
    plt.show()
    # animation.save('model.gif', writer='imagemagick', fps=3)
    plt.close()