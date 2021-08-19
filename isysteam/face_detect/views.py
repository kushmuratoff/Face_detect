from django.shortcuts import render
from .models import Person,Vaqti
def home(request):

    # return render  (request, '/admin/')
    # p = Person(Ism="Person")
    # p.save()
    # print(p.id)
    # p.Ism=p.Ism+str(p.id)
    # p.save()
    # print(p.Ism)

    # face_match(,,p.Ism)

    # Pid = Person.objects.all().order_by('id')
    # print(len(Pid))
    Xisobot = Vaqti.objects.all()
    print(Xisobot, "dfsd")




    return render(request, 'face_detect/home.html',{'Xisobot':Xisobot})