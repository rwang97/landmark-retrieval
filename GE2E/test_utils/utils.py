import numpy as np
import sys
import csv

def MeanAveragePrecision(predictions, retrieval_solution, max_predictions=100):
  """Computes mean average precision for retrieval prediction.
  Args:
    predictions: Dict mapping test image ID to a list of strings corresponding
      to index image IDs.
    retrieval_solution: Dict mapping test image ID to list of ground-truth image
      IDs.
    max_predictions: Maximum number of predictions per query to take into
      account. For the Google Landmark Retrieval challenge, this should be set
      to 100.
  Returns:
    mean_ap: Mean average precision score (float).
  Raises:
    ValueError: If a test image in `predictions` is not included in
      `retrieval_solutions`.
  """
  # Compute number of test images.
  num_test_images = len(retrieval_solution.keys())

  # Loop over predictions for each query and compute mAP.
  mean_ap = 0.0
  for key, prediction in predictions.items():
    if key not in retrieval_solution:
      raise ValueError('Test image %s is not part of retrieval_solution' % key)

    # Loop over predicted images, keeping track of those which were already
    # used (duplicates are skipped).
    ap = 0.0
    already_predicted = set()
    num_expected_retrieved = min(len(retrieval_solution[key]), max_predictions)
    num_correct = 0
    for i in range(min(len(prediction), max_predictions)):
      if prediction[i] not in already_predicted:
        if prediction[i] in retrieval_solution[key]:
          num_correct += 1
          ap += num_correct / (i + 1)
        already_predicted.add(prediction[i])

    ap /= num_expected_retrieved
    mean_ap += ap

  mean_ap /= num_test_images

  return mean_ap


def MeanPrecisions(predictions, retrieval_solution, max_predictions=100):
  """Computes mean precisions for retrieval prediction.
  Args:
    predictions: Dict mapping test image ID to a list of strings corresponding
      to index image IDs.
    retrieval_solution: Dict mapping test image ID to list of ground-truth image
      IDs.
    max_predictions: Maximum number of predictions per query to take into
      account.
  Returns:
    mean_precisions: NumPy array with mean precisions at ranks 1 through
      `max_predictions`.
  Raises:
    ValueError: If a test image in `predictions` is not included in
      `retrieval_solutions`.
  """
  # Compute number of test images.
  num_test_images = len(retrieval_solution.keys())

  # Loop over predictions for each query and compute precisions@k.
  precisions = np.zeros((num_test_images, max_predictions))
  count_test_images = 0
  for key, prediction in predictions.items():
    if key not in retrieval_solution:
      raise ValueError('Test image %s is not part of retrieval_solution' % key)

    # Loop over predicted images, keeping track of those which were already
    # used (duplicates are skipped).
    already_predicted = set()
    num_correct = 0
    for i in range(max_predictions):
      if i < len(prediction):
        if prediction[i] not in already_predicted:
          if prediction[i] in retrieval_solution[key]:
            num_correct += 1
          already_predicted.add(prediction[i])
      precisions[count_test_images, i] = num_correct / (i + 1)
    count_test_images += 1

  mean_precisions = np.mean(precisions, axis=0)

  return mean_precisions


def ReadSolution(file_path):
  """Reads solution from file, for a given task.
  Args:
    file_path: Path to CSV file with solution. File contains a header.
    task: Type of challenge task. Supported values: 'recognition', 'retrieval'.
  Returns:
    public_solution: Dict mapping test image ID to list of ground-truth IDs, for
      the Public subset of test images. If `task` == 'recognition', the IDs are
      integers corresponding to landmark IDs. If `task` == 'retrieval', the IDs
      are strings corresponding to index image IDs.
    private_solution: Same as `public_solution`, but for the private subset of
      test images.
    ignored_ids: List of test images that are ignored in scoring.
  Raises:
    ValueError: If Usage field is not Public, Private or Ignored; or if `task`
      is not supported.
  """
  public_solution = {}
  private_solution = {}
  ignored_ids = []
  with open(file_path, 'r') as csv_file:
    reader = csv.reader(csv_file)
    next(reader, None)  # Skip header.
    for row in reader:
      test_id = row[0]
      if row[2] == 'Ignored':
        ignored_ids.append(test_id)
      else:
        ground_truth_ids = []

        for image_id in row[1].split(' '):
            ground_truth_ids.append(image_id)

        if row[2] == 'Public':
          public_solution[test_id] = ground_truth_ids
        elif row[2] == 'Private':
          private_solution[test_id] = ground_truth_ids
        else:
          raise ValueError('Test image %s has unrecognized Usage tag %s' %
                           (row[0], row[2]))

  return public_solution, private_solution, ignored_ids


def ReadPredictions(file_path, public_ids, private_ids, ignored_ids):
  """Reads predictions from file, for a given task.
  Args:
    file_path: Path to CSV file with predictions. File contains a header.
    public_ids: Set (or list) of test image IDs in Public subset of test images.
    private_ids: Same as `public_ids`, but for the private subset of test
      images.
    ignored_ids: Set (or list) of test image IDs that are ignored in scoring and
      are associated to no ground-truth.
    task: Type of challenge task. Supported values: 'recognition', 'retrieval'.
  Returns:
    public_predictions: Dict mapping test image ID to prediction, for the Public
      subset of test images. If `task` == 'recognition', the prediction is a
      dict with keys 'class' (integer) and 'score' (float). If `task` ==
      'retrieval', the prediction is a list of strings corresponding to index
      image IDs.
    private_predictions: Same as `public_predictions`, but for the private
      subset of test images.
  Raises:
    ValueError:
      - If test image ID is unrecognized/repeated;
      - If `task` is not supported;
      - If prediction is malformed.
  """
  public_predictions = {}
  private_predictions = {}
  with open(file_path, 'r') as csv_file:
    reader = csv.reader(csv_file)
    next(reader, None)  # Skip header.
    for row in reader:
      # Skip row if empty.
      if not row:
        continue

      test_id = row[0]

      # Makes sure this query has not yet been seen.
      if test_id in public_predictions:
        raise ValueError('Test image %s is repeated.' % test_id)
      if test_id in private_predictions:
        raise ValueError('Test image %s is repeated' % test_id)

      # If ignored, skip it.
      if test_id in ignored_ids:
        continue

      # Only parse result if there is a prediction.
      if row[1]:
        prediction_split = row[1].split(' ')
        # Remove empty spaces at end (if any).
        if not prediction_split[-1]:
          prediction_split = prediction_split[:-1]

        prediction_entry = prediction_split

        if test_id in public_ids:
          public_predictions[test_id] = prediction_entry
        elif test_id in private_ids:
          private_predictions[test_id] = prediction_entry
        else:
          raise ValueError('test_id %s is unrecognized' % test_id)

  return public_predictions, private_predictions