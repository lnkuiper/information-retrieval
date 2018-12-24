# Ad-hoc plotting for report

import matplotlib.pyplot as plt
from experiments import get_results_from_dir
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.patheffects as path_effects

def plot_dir(result_folder):
	results = get_results_from_dir(result_folder)
	results_combi = sorted(zip(results['mu'],results['P_30'],results['map']))
	plt.plot([result[0] for result in results_combi],[result[1] for result in results_combi])
	plt.plot([result[0] for result in results_combi],[result[2] for result in results_combi])
	plt.xlabel('μ')
	plt.ylabel('score')
	plt.title('Dirichlet',{'family' : 'normal','size'   : 16})
	big_map = 0
	map_i = -1
	big_p = 0
	p_i = -1
	for i, result in enumerate(results_combi):
		if result[2] > big_map:
			map_i = result[0]
			big_map = result[2]
		if result[1] > big_p:
			p_i = result[0]
			big_p = result[1]
	plt.scatter(p_i,big_p)
	plt.scatter(map_i,big_map)

	text_font = {'family' : 'normal','size'   : 13}
	plt.text(map_i+0.005,big_map+0.003,'μ=800', text_font)
	plt.text(p_i+0.005,big_p+0.003,'μ=900', text_font)
	plt.legend(['map','P30'])

	print(big_map, map_i)
	print(big_p,p_i)
	plt.ylim([0.225,0.315])
	plt.show()


def plot_jm(result_folder):
	results = get_results_from_dir(result_folder)
	results_combi = sorted(zip(results['lambda'],results['P_30'],results['map']))
	plt.plot([result[0] for result in results_combi],[result[1] for result in results_combi])
	plt.plot([result[0] for result in results_combi],[result[2] for result in results_combi])
	plt.xlabel('λ')
	plt.ylabel('score')
	plt.title('Jelinek-Mercer',{'family' : 'normal','size'   : 24})
	big_map = 0
	map_i = -1
	big_p = 0
	p_i = -1
	for i, result in enumerate(results_combi):
		if result[2] > big_map:
			map_i = result[0]
			big_map = result[2]
		if result[1] > big_p:
			p_i = result[0]
			big_p = result[1]
	plt.scatter(p_i,big_p)
	plt.scatter(map_i,big_map)

	text_font = {'family' : 'normal','size'   : 13}
	plt.text(map_i+0.005,big_map+0.003,'λ=0.275', text_font)
	plt.text(p_i+0.005,big_p+0.003,'λ=0.2', text_font)
	plt.legend(['P30','map'])

	print(big_map, map_i)
	print(big_p,p_i)
	plt.ylim([0.15,0.3])
	plt.show()


def plot_bm25(result_folder):
	results = get_results_from_dir(result_folder)
	print(max(results['P_30']),max(results['map']))
	results_combi = sorted(zip(results['k1'],results['b'], results['P_30'], results['map']))
	print(len(results_combi))
	X, Y = np.meshgrid(np.asarray(sorted(set([result[1] for result in results_combi]))),np.asarray(sorted(set([result[0] for result in results_combi]))))

	big_map = 0
	map_i = -1
	big_p = 0
	p_i = -1
	for i, result in enumerate(results_combi):
		if result[3] > big_map:
			map_i = (result[0], result[1])
			big_map = result[3]
		if result[2] > big_p:
			p_i = (result[0], result[1])
			big_p = result[2]
	print('Highest MAP: ' + str(big_map) + ', with params: ', map_i)
	print('Highest P30: ' + str(big_p) + ', with params: ', p_i)

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	plt.hold(True)
	# ax.plot((map_i[1], map_i[1]), (map_i[0], map_i[0]), zs=(0.2447, 0.2449), linewidth=5, color='black', zorder=0, solid_capstyle='round')
	ax.plot((p_i[1], p_i[1]), (p_i[0], p_i[0]), zs=(0.3011, 0.3013), linewidth=5, color='black', zorder=0, solid_capstyle='round')
	# text = ax.text(map_i[1], map_i[0], big_map+0.003, 'b={:.2f}, $k_1$={:.3f}'.format(map_i[1],map_i[0]), size=10, color='white')
	text = ax.text(p_i[1], p_i[0], big_p+0.003, 'b={:.2f}, $k_1$={:.3f}'.format(p_i[1],p_i[0]), size=10, color='white')
	text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])
	# ax.plot_surface(X, Y, np.reshape(np.asarray([result[3] for result in results_combi]),(16,20)), cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=1, zorder=100)
	ax.plot_surface(X, Y, np.reshape(np.asarray([result[2] for result in results_combi]),(16,20)), cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=1, zorder=100)
	plt.title('BM25')
	ax.set_xlabel('b')
	ax.set_ylabel('$k_1$')
	ax.set_zlabel('score')
	plt.show()

if __name__ == '__main__':
	#plot_jm('../results/jm/')
	#plot_dir('../results/dir/')
	plot_bm25('../results/bm25/')

	exit()
