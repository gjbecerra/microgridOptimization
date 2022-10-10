%% CREG 15/2018 - Analysis of pricing equations
fch_vec = 2:-0.01:1;
for ind = 1:length(fch_vec)
    P = 0.01*[60,65,70,70,75,80,80,80,80,80,75,75,75,80,85,90,90,95,100,100,95,85,80,65];
    % fch = 2;
    fch = fch_vec(ind);
    indx = (P >= 0.95);
    indz = (P < 0.95) & (P >= 0.75);
    indy = (P < 0.75);
    Hx = sum(indx);
    Hz = sum(indz);
    Hy = sum(indy);
    Px = mean(P(indx));
    Pz = mean(P(indz));
    Py = mean(P(indy));
    Dt = 350;
    
    %
    syms Dx Dz Dy
    Soln = solve(Hx*Px*Dx/fch + Hz*Pz*Dz + fch*Hy*Py*Dy == Dt*sum(P),...
        Dx/(fch*Dz) == Px/Pz,...
        Dx/(fch^2*Dy) == Px/Py);
    Dx = double(Soln.Dx);
    Dz = double(Soln.Dz);
    Dy = double(Soln.Dy);
    Cv = sum(Dx*P(indx)) + sum(Dz*P(indz)) + sum(Dy*P(indy));
    Cf = sum(Dt*P);
    
    %
    f1_Dx = @(Dy,Dz) fch*(Dt*sum(P) - Hz*Pz*Dz - fch*Hy*Py*Dy)/Hx*Px;
    f2_Dx = @(Dy,Dz) (Px*fch*Dz)./Pz;
    f3_Dx = @(Dy,Dz) (Px*fch^2*Dy)./Py;
    [Y,Z] = meshgrid(0:100:1000,0:100:1000);
    X1 = f1_Dx(Y,Z);
    X2 = f2_Dx(Y,Z);
    X3 = f3_Dx(Y,Z);
    figure(1), cla, hold on
    xlabel('Dy'), ylabel('Dz'), zlabel('Dx')
    surf(Y,Z,X1,'FaceColor','red','FaceAlpha',0.5)
    surf(Y,Z,X2,'FaceColor','green','FaceAlpha',0.5)
    surf(Y,Z,X3,'FaceColor','blue','FaceAlpha',0.5)
    scatter3(Dy,Dz,Dx,...
        'MarkerEdgeColor','k',...
        'MarkerFaceColor',[0.75 0 0])
    axis([0 1000 0 1000 0 1000])
    hold off
    view(130,30)
    drawnow
%     pause(0.1)
    title(sprintf("Dx = %s, Dz = %s, Dy = %s, Cv = %s, Cf = %s",...
          num2str(Dx),num2str(Dz),num2str(Dy),num2str(Cv),num2str(Cf)))
end