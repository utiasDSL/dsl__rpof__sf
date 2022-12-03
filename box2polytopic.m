% box2polytopic
%
% Technical University of Munich
% University of Toronto Institute for Aerospace Studies
% Learning Systems and Robotics Lab
%
% Author
% Lukas Brunke: lukas.brunke@tum.de

function [L, l] = box2polytopic(Z_limits, dim)
    % Turns box constraints to polytopic constraints
    
    % This assumes that constraints contain the origin
    L = [];
    l = [];

    eye_dim = eye(dim);

    for constraint_id = 1 : size(Z_limits, 1)
        entry_id = Z_limits(constraint_id, 1);
        if Z_limits(constraint_id, 2) ~= -inf
            if Z_limits(constraint_id, 2) == 0
                l = [l; 0];
                L = [L; - eye_dim(constraint_id, :)];
            else
                % Note: negative sign is in L matrix 
                l = [l; 1];
                factor = 1 / Z_limits(constraint_id, 2);
                L = [L; factor * eye_dim(entry_id, :)];
            end
        end
        if Z_limits(constraint_id, 3) ~= inf
            if Z_limits(constraint_id, 3) == 0
                l = [l; 0];
                L = [L; eye_dim(constraint_id, :)];
            else
                l = [l; 1];
                factor = 1 / Z_limits(constraint_id, 3);
                L = [L; factor * eye_dim(entry_id, :)];
            end
        end
    end
end