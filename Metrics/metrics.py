import numpy as np
import sys
import utils


def metrics(solution_path, predictions_path):

  # Read solution.
  print('Reading solution...')
  public_solution, private_solution, ignored_ids = utils.ReadSolution(
      solution_path)
  print('done!')

  # Read predictions.
  print('Reading predictions...')
  public_predictions, private_predictions = utils.ReadPredictions(
    predictions_path, set(public_solution.keys()),
    set(private_solution.keys()), set(ignored_ids))
  print('done!')

  # Mean average precision.
  print('**********************************************')
  print('(Public)  Mean Average Precision: %f' %
        utils.MeanAveragePrecision(public_predictions, public_solution))
  print('(Private) Mean Average Precision: %f' %
        utils.MeanAveragePrecision(private_predictions, private_solution))

  # Mean precision@k.
  print('**********************************************')
  public_precisions = 100.0 * utils.MeanPrecisions(public_predictions,
                                                     public_solution)
  private_precisions = 100.0 * utils.MeanPrecisions(private_predictions,
                                                      private_solution)
  print('(Public)  Mean precisions: P@1: %.2f, P@5: %.2f, P@10: %.2f, '
        'P@50: %.2f, P@100: %.2f' %
        (public_precisions[0], public_precisions[4], public_precisions[9],
         public_precisions[49], public_precisions[99]))
  print('(Private) Mean precisions: P@1: %.2f, P@5: %.2f, P@10: %.2f, '
        'P@50: %.2f, P@100: %.2f' %
        (private_precisions[0], private_precisions[4], private_precisions[9],
         private_precisions[49], private_precisions[99]))
