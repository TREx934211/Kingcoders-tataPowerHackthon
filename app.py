from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__, template_folder='src')

# Load your trained TensorFlow model
model = tf.keras.models.load_model('src/model')

Para = ['temperature_2_m_above_gnd',
        	'relative_humidity_2_m_above_gnd',	
            'mean_sea_level_pressure_MSL',	
            'total_precipitation_sfc',	
            'snowfall_amount_sfc',	
            'total_cloud_cover_sfc',	
            'high_cloud_cover_high_cld_lay',	
            'medium_cloud_cover_mid_cld_lay',	
            'low_cloud_cover_low_cld_lay',
            'shortwave_radiation_backwards_sfc',	
            'wind_speed_10_m_above_gnd',	
            'wind_direction_10_m_above_gnd',	
            'wind_speed_80_m_above_gnd',	
            'wind_direction_80_m_above_gnd',	
            'wind_speed_900_mb',	
            'wind_direction_900_mb',	
            'wind_gust_10_m_above_gnd',	
            'angle_of_incidence',	
            'zenith',	
            'cazimuth']

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        input_data = [float(request.form[f'input_{i}']) for i in range(20)]
        input_data = np.array(input_data).reshape(1, 20)  # Reshape data for prediction
        prediction = model.predict(input_data)[0][0]
        if ( prediction < 0 ):
            prediction = 0
        No_of_plates = prediction * 5.15 
        Area = prediction * 5.15 
        return render_template('result.html', prediction=prediction, NOP=No_of_plates, Area=Area)
    return render_template('input_form.html', Para=Para)

if __name__ == '__main__':
    app.run(debug=True)
