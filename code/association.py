import numpy as np

# Define landmark id given appearance
# Input:
#   appearance: appearance of given landmark
#   landmarks: list of landmarks appearances
#   num_landmarks: total number of landmarks
# Output:
#   index: index of the landmark
def landmarkAssociation(appearance, land_appearances):

    observ_appearance = np.asarray(appearance).reshape([-1, 1])

    diff = land_appearances - observ_appearance
    error = np.linalg.norm(diff, ord=2, axis=0)
    index = np.argmin(error)
    return index