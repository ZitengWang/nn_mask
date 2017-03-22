function mat2json(mat,filename)

% MAT2JSON Writes a JSON file
%
% mat2json(mat,filename)
%
% Inputs:
% mat: Matlab cell array whose entries are Matlab structures containing the
% value for each JSON field
% filename: JSON filename (.json extension)
%
% Note: using JSON2MAT followed by MAT2JSON will generally not lead back to
% the original JSON file due to the loss of digits beyond double precision
% and to the handling of trailing zeros.
%
% If you use this software in a publication, please cite:
%
% Jon Barker, Ricard Marxer, Emmanuel Vincent, and Shinji Watanabe, The
% third 'CHiME' Speech Separation and Recognition Challenge: Dataset,
% task and baselines, submitted to IEEE 2015 Automatic Speech Recognition
% and Understanding Workshop (ASRU), 2015.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2015 University of Sheffield (Jon Barker, Ricard Marxer)
%                Inria (Emmanuel Vincent)
%                Mitsubishi Electric Research Labs (Shinji Watanabe)
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fid=fopen(filename,'w');
fprintf(fid,'%s\n','[');
for ind=1:length(mat), % loop over entries
    fprintf(fid,'    %s\n','{'); % entry delimiter
    fields=fieldnames(mat{ind});
    for f=1:length(fields), % loop over fields
        field=fields{f};
        value=mat{ind}.(field);
        if ischar(value), % text field
            fprintf(fid,'        "%s": "%s"',field,value);
        elseif islogical(value), % boolean field
            if value,
                fprintf(fid,'        "%s": true',field);
            else
                fprintf(fid,'        "%s": false',field);
            end
        elseif value==floor(value), % integer field
            fprintf(fid,'        "%s": %d',field,value);
        else % double field
            fprintf(fid,'        "%s": %17.*f',field,15-max(0,floor(log10(value))),value);
        end
        if f~=length(fields), % field delimiter
            fprintf(fid,', ');
        end
        fprintf(fid,'\n');
    end
    fprintf(fid,'    }'); % entry delimiter
    if ind~=length(mat),
        fprintf(fid,', ');
    end
    fprintf(fid,'\n');
end
fprintf(fid,']');
fclose(fid);

return