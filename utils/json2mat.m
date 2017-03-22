function mat=json2mat(filename)

% JSON2MAT Reads a JSON file
%
% mat=json2mat(filename)
%
% Input:
% filename: JSON filename (.json extension)
%
% Output:
% mat: Matlab cell array whose entries are Matlab structures containing the
% value for each JSON field
%
% Note: all numeric fields are rounded to double precision. Digits beyond
% double precision are lost.
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

fid=fopen(filename,'r');
fgetl(fid); % [
txt=fgetl(fid); % { or ]
txt=fgetl(fid); % first field
mat=cell(1);
ind=1; % entry index
while txt~=-1, % end of file
    if strcmp(txt,'    }, ') || strcmp(txt,'    }'), % next entry
        ind=ind+1;
        txt=fgetl(fid); % { or ]
    else
        try
        pos=strfind(txt,'"');
        field=txt(pos(1)+1:pos(2)-1);
        catch
            keyboard;
        end
        if ~strcmp(txt(end-1:end),', '), % last field
            txt=txt(pos(2)+3:end);
        else
            txt=txt(pos(2)+3:end-2);
        end
        if strcmp(txt(1),'"') && strcmp(txt(end),'"'), % text value
            value=txt(2:end-1);
        else % boolean or numerical value
            value=eval(txt);
        end
        mat{ind}.(field)=value;
    end
    txt=fgetl(fid); % next field
end
fclose(fid);

return