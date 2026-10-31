"""Generate original watertight OBJ meshes in metres, along local X."""
from pathlib import Path
import os,tempfile
import numpy as np

def write_obj(path,v,f):
    payload=('# Procedural pen mesh, metres; generated for this project\n'+''.join('v %.9f %.9f %.9f\n'%tuple(p) for p in v)+''.join('f %d %d %d\n'%tuple(i+1 for i in t) for t in f)).encode('utf-8')
    if path.exists() and path.read_bytes()==payload:return
    # Parallel MuJoCo workers must never observe a truncated OBJ during generation.
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as handle:
        handle.write(payload);temporary=Path(handle.name)
    try:
        if path.exists() and path.read_bytes()==payload:temporary.unlink();return
        try:os.replace(temporary,path)
        except PermissionError:
            if not path.exists() or path.read_bytes()!=payload:raise
    finally:
        if temporary.exists():temporary.unlink()

def generate():
    dest=Path(__file__).parent/'assets/pen'; dest.mkdir(parents=True,exist_ok=True)
    n=96; v=[]; f=[]
    for x in [-.089,.089]:
        v.extend((x,.00745*np.cos(a),.00745*np.sin(a)) for a in np.arange(n)*2*np.pi/n)
    v.extend([(-.089,0,0),(.089,0,0)])
    for i in range(n):
        j=(i+1)%n
        f.extend([(i,j,n+j),(i,n+j,n+i),(2*n,j,i),(2*n+1,n+i,n+j)])
    write_obj(dest/'shaft.obj',v,f)
    v=[(x,y,z) for x in [-.018,.018] for y in [-.009,.009] for z in [-.009,.009]]
    faces=[(0,1,3),(0,3,2),(4,6,7),(4,7,5),(0,4,5),(0,5,1),(2,3,7),(2,7,6),(0,2,6),(0,6,4),(1,5,7),(1,7,3)]
    write_obj(dest/'endcap.obj',v,faces)
    return dest

if __name__=='__main__':print(generate())
