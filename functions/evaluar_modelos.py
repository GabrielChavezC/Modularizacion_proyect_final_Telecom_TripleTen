import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

# Función para probar los modelos y graficar la curva ROC
def evaluate_model(model, train_features, train_target, test_features, test_target, save_path=''):
    eval_stats = {}

    fig, axs = plt.subplots(1, 1, figsize=(10, 6))  # Solo un subplot para la curva ROC

    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):
        eval_stats[type] = {}

        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]

        # ROC
        fpr, tpr, _ = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)
        eval_stats[type]['ROC AUC'] = roc_auc

        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # ROC
        ax = axs
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower right')
        ax.set_title(f'Curva ROC')

        eval_stats[type]['Accuracy'] = metrics.accuracy_score(target, pred_target)

    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(3)
    df_eval_stats = df_eval_stats.reindex(index=('Accuracy', 'ROC AUC'))

    print(df_eval_stats)

    # Guardar la imagen
    plt.savefig(save_path)
    #plt.show()
    return
