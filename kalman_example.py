import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import random
def init_bar():
    truth_plot = plt.bar(x_array + 0.1, the_truth, color = 'blue', alpha = 0.2)
    bar_plot = plt.bar(x_array, model_data, color = 'pink')
    plt.gca().tick_params(axis = 'both', labelsize = 16)
    plt.title(f'Time step 0', fontsize = 26)
    plt.ylim((0, 3.2))
    plt.xticks(x_array)
    plt.grid(True)
    return truth_plot, bar_plot

def init_model_matrix():
    model_matrix = 2 * np.zeros((model_size, model_size))
    for j in range(model_size - 1):
        model_matrix[j,j+1] = 1
    model_matrix[model_size - 1,0] = 1
    return model_matrix


if __name__ == '__main__':
    np.set_printoptions(precision=1)
    model_size   = 9
    x_array      = np.linspace(1, model_size, num = model_size, endpoint = True)
    the_truth    = 0.5 * np.cos(x_array * 2 * np.pi / model_size) + 2.5
    
    obs_variance = 0.5

    model_data   = 0.5 * np.sin(x_array * 2 * np.pi / model_size) + 1.5
    model_cov    = 0.5 * np.eye(model_size, model_size)
    model_matrix = init_model_matrix()

    reanalysis_cov = np.zeros((model_size, model_size))

    fig      = plt.figure(figsize=(19.2, 10.8), dpi = 50)
    truth_plot, bar_plot = init_bar()

    central_idx = 4
    bar_plot[central_idx].set_color('red')
    def update(frame):
        global model_data, the_truth, reanalysis_cov, central_idx
        model_data = np.dot(model_matrix, model_data)
        the_truth = np.roll(the_truth, -1)

        obs_size     = random.randint(0, 5)
        observations = np.zeros((obs_size,))
        obs_model    = np.zeros((obs_size, model_size))

        for bar, val in zip(truth_plot, the_truth):
            bar.set_height(val)
            bar.set_alpha(0.2)

        for ob in range(obs_size):
            idx = random.randint(0, model_size - 1)
            observations[ob]   = the_truth[idx] + np.random.normal(0, 0.5)
            obs_model[ob, idx] = 1
            truth_plot[idx].set_alpha(1)

        obs_cov = obs_variance * np.eye(obs_size, obs_size)

        predict_cov = np.dot(model_matrix, reanalysis_cov)
        predict_cov = np.dot(predict_cov, np.transpose(model_matrix))
        predict_cov += model_cov

        innov_mat   = np.dot(np.dot(obs_model, predict_cov), np.transpose(obs_model))
        innov_mat   += obs_cov

        kalman_gain = np.dot(predict_cov, np.transpose(obs_model))
        kalman_gain = np.dot(kalman_gain, np.linalg.inv(innov_mat))

        model_data += np.dot(kalman_gain, observations - np.dot(obs_model, model_data))
        reanalysis_cov = np.dot(np.eye(model_size, model_size) - np.dot(kalman_gain, obs_model), predict_cov)
        print(reanalysis_cov)

        plt.title(f'Time step {frame+1}', fontsize = 26)
        for bar, val in zip(bar_plot, model_data):
            bar.set_height(val)
        bar_plot[central_idx].set_color('pink')
        central_idx = (central_idx - 1) % model_size
        bar_plot[central_idx].set_color('red')
    
    animation = anim.FuncAnimation(fig, update, frames = model_size * 10, repeat = False, interval = 500)
    plt.show()
    plt.close()