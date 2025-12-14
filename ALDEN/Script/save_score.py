"""
Utility functions for saving model prediction scores to files.
"""
def save_scores(cm_score, file_list, label_list, save_path):
    """
    Save model prediction results to a score file.
    
    Args:
        cm_score: Model prediction scores, format: np.array
        file_list: Sample names, format: np.array
        label_list: Ground truth labels, format: np.array
        save_path: Path to save the score file
    """
    with open(save_path, 'w+') as fh:
        for s, utt, label in zip(cm_score, file_list, label_list):
            fh.write('{} {} {:.4f}\n'.format(utt, label, s))
    fh.close()
    print('Scores saved to : {}'.format(save_path))
