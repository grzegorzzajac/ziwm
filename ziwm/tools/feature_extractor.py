import numpy as np
from sklearn import preprocessing

def OHE_encode(feature_vect):
    enc = preprocessing.OneHotEncoder()
    le = preprocessing.LabelEncoder()
    le.fit(feature_vect)
    vect_classes = le.transform(feature_vect)
    enc.fit(vect_classes)
    vect_OHE = enc.transform(vect_classes).toarray()
    return vect_OHE

def encode_class(column_vect):
    le = preprocessing.LabelEncoder()
    le.fit(column_vect)
    class_vect = le.transform(column_vect)
    return class_vect

def scale(column_vect):
    scaler = preprocessing.MinMaxScaler()
    feature_vect = scaler.fit_transform(column_vect)
    return feature_vect

def load_file(filename):
    file_data = np.genfromtxt(filename, dtype='str', delimiter=',')
    column_types = file_data[0]
    raw_data = file_data[1:]

    return raw_data, column_types

def save_to_file(filename, data, header):
    hdr = None
    if header is not None:
        hdr = ','.join([str(x) for x in header])
    np.savetxt(filename, data, header=hdr)
    
def create_features(raw_data, column_types):
    column_count = raw_data.shape[1]
    assert column_count == column_types.size
    
    X_data = None
    Y = None
    data_feature_indices = [-1]     # -1 for y
    feature_num = 0
    
    for column_index in range(column_count):
        col_type = column_types[column_index]

        # is a category type
        if col_type == 'c':

            column_arr = raw_data[:, column_index].reshape(-1, 1)
            columns_OHE = OHE_encode(column_arr)
            columns_OHE_count = columns_OHE.shape[1]
            if X_data is None:
                X_data = columns_OHE
            else:
                X_data = np.hstack((X_data, columns_OHE))

            feature_indices = [feature_num] * columns_OHE_count
            feature_num += 1
            data_feature_indices.extend(feature_indices)

        # is a real type
        elif col_type == 'r':

            column_arr = raw_data[:, column_index].astype(float).reshape(-1, 1)
            column_scaled = scale(column_arr)
            if X_data is None:
                X_data = column_scaled
            else:
                X_data = np.hstack((X_data, column_scaled))

            data_feature_indices.append(feature_num)
            feature_num += 1

        # is an output class
        elif col_type == 'y':

            if Y is not None:
                raise Exception("'y' column already encountered")
            else:
                Y_raw = raw_data[:,column_index].reshape(-1, 1)
                Y = encode_class(Y_raw)
                
    dataset = np.hstack((Y, X_data))
    return dataset, data_feature_indices
