% in_limits
%
% Technical University of Munich
% University of Toronto Institute for Aerospace Studies
% Learning Systems and Robotics Lab
%
% Author
% Lukas Brunke: lukas.brunke@tum.de

function inside_limits = in_limits(x, limits)
    % Checks if x is inside the specified limits
    num_limits = size(limits, 1);
    for i = 1 : num_limits
        if x(i) < limits(i, 2) || x(i) > limits(i, 3)
            inside_limits = false;
            return;
        end
    end
    inside_limits = true;
end
