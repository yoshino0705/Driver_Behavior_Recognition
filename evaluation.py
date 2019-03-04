# Warning: This code actually ran for several hours because there were 9 models 
# trained and tested on 51 combinations of the original dataset
#
# Do not attempt to execute the script unless you're fine with leaving the computer running for several hours
# You have been warned!

from driver_behavior import DriverBehavior

def save_to_file(text='test'):
    with open('stats.txt','a') as f:
        f.write('\n' + text)

if __name__ == '__main__':
    path = 'data.csv'
    db = DriverBehavior(path)
    
    # these are sorted based on feature_importances.csv
    # you could try run the calculations yourself using the links provided in README.md to confirm
    top_features = ['Engine_coolant_temperature.1', 'Engine_soacking_time', 'Torque_of_friction', 'Long_Term_Fuel_Trim_Bank1', 'Maximum_indicated_engine_torque', 'Engine_coolant_temperature', 'Intake_air_pressure', 'Calculated_road_gradient', 'Steering_wheel_angle', 'Master_cylinder_pressure', 'Engine_speed', 'Accelerator_Pedal_value', 'Fuel_consumption', 'Current_spark_timing', 'Acceleration_speed_-_Lateral', 'Engine_Idel_Target_Speed', 'Throttle_position_signal', 'Calculated_LOAD_value', 'Wheel_velocity_front_left-hand', 'Current_Gear', 'Acceleration_speed_-_Longitudinal', 'Absolute_throttle_position', 'Short_Term_Fuel_Trim_Bank1', 'Engine_torque_after_correction', 'Wheel_velocity_rear_right-hand', 'Flywheel_torque', 'Wheel_velocity_front_right-hand', 'Engine_torque', 'Torque_converter_speed', 'Flywheel_torque_(after_torque_interventions)', 'Wheel_velocity_rear_left-hand', 'Steering_wheel_speed', 'Activation_of_Air_compressor', 'Torque_converter_turbine_speed_-_Unfiltered', 'Vehicle_speed', 'Minimum_indicated_engine_torque', 'Indication_of_brake_switch_ON/OFF', 'Gear_Selection', 'Converter_clutch', 'TCU_requested_engine_RPM_increase', 'TCU_requests_engine_torque_limit_(ETL)', 'Clutch_operation_acknowledge', 'Engine_in_fuel_cut_off', 'Inhibition_of_engine_fuel_cut_off', 'Target_engine_speed_used_in_lock-up_module', 'Glow_plug_control_request', 'Torque_scaling_factor(standardization)', 'Requested_spark_retard_angle_from_TCU', 'Filtered_Accelerator_Pedal_value', 'Fuel_Pressure', 'Standard_Torque_Ratio']
    
    indices_map = {k:v for v,k in enumerate(list(db.X))}
    results = {}
    
    # the madness begins
    for n in range(2, 52):
    print('Currently on iteration N = ', n)
    for name in db.models.keys():
        print('Training with', name)
        selected_indices = [indices_map[k] for k in top_features[:n]]
        
        db.train(name, selected_features=selected_indices)
    print('Testing train_accuracy on all trained models')
    train_acc = db.train_accuracy(force_update=True, selected_features=selected_indices)
    print('Testing test_accuracy on all trained models')
    test_acc = db.test_accuracy(force_update=True, selected_features=selected_indices)
    results[n] = {'train':train_acc, 'test':test_acc}
    save_to_file('{}: {}'.format(n , results[n]))
    
