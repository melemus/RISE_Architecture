import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('maddpg.csv')
plt.figure()
plt.plot(df.loc[:, df.columns[0]], label='Training Scores', linestyle='solid', color='black')
plt.plot(df.loc[:, df.columns[-1]], label='Testing Scores', linestyle=':', color='black')
plt.xlabel('Episodes', fontsize=12)
plt.ylabel('Total Reward', fontsize=12)
plt.title('Training and Testing Performance', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(False)
plt.savefig('Plots/train_test_scores.png')
plt.show()

df2 = pd.read_csv('each_agent.csv')
plt.plot(df2.loc[:, df2.columns[0]], label='Agent 1 - Training', linestyle='-', color='black', linewidth=1)
plt.plot(df2.loc[:, df2.columns[3]], label='Agent 1 - Testing', linestyle=':', color='black',linewidth=1)

plt.plot(df2.loc[:, df2.columns[1]], label='Agent 2 - Training', color='#495057',linestyle='-',  marker='.',markevery=20,linewidth=1)
plt.plot(df2.loc[:, df2.columns[4]], label='Agent 2 - Testing', linestyle=':', color='#495057', marker='.',markevery=20,linewidth=1)

plt.plot(df2.loc[:, df2.columns[2]], label='Agent 3 - Training', color='#6c757d',linestyle='-', marker='x',markevery=20,linewidth=1)
plt.plot(df2.loc[:, df2.columns[-1]], label='Agent 3 - Testing', linestyle=':', color='#6c757d', marker='x',markevery=20,linewidth=1)

plt.legend(loc='center', fontsize=10, ncol=2, frameon=False) 
plt.xlabel('Episodes', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.title('Training vs Testing Rewards for Each Agent', fontsize=12)
plt.grid(False)  
# plt.tight_layout()  
plt.savefig('Plots/train_test_comparison_per_agent.png')
plt.show()