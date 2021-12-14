import tkinter.filedialog as filedialog
from tkinter import ttk
import tkinter as tk
import os
import configparser
import re
from shutil import copyfile
import parameter_extraction

def open_file(arg,dire):
    file = filedialog.askopenfilename(initialdir = dire,title = "Select file",filetypes = (("all files","*.*"),("tif files","*.tif"),("cfg files","*.cfg")))
    arg.delete(0,tk.END)
    arg.insert(0,file)

def get_variables(dire,con_str,acc_str,don_str,range_str,res_str,par_state,nwindow_str,eps_entry_str,crop_str,register_state\
                  ,union_state,acc_bleach_str,don_bleach_str,bleach_fit,tiff_state,h5_state,anim_state,option_dict,option_line):

    # Run options and verbose flag
    option = int(option_dict[option_line.get()])
    verbose = True

    if len(con_str.get()) == 0:
        # Fill in GUI without config file
        assert(len(acc_str.get()) > 0), "Acceptor filepath must exist"

        # Input_path, filename and extension of acceptor channel
        acc = acc_str.get()
        acc_posa = acc.find("acceptor")
        acc_posl = [m.start() for m in re.finditer('/', acc[:acc_posa])]
        acc_posd = acc.find(".")

        input_path = acc[:acc_posl[-1]]
        filename = acc[acc_posl[-1]+1:acc_posa-1]
        extension = acc[acc_posd+1:]

        # if donor path exists, find the input path of the donor channel
        if len(don_str.get()) > 0:
            don = don_str.get()
            don_posa = don.find("acceptor")
            don_posl = [m.start() for m in re.finditer('/', don[:don_posa])]
            don_input_path = don[:don_posl[-1]]
        else:
            don_input_path = ''

        # Check the acceptor and donor stacks based on the run option
        assert(input_path == don_input_path or len(don_input_path) == 0), "Acceptor and Donor stacks must be in the same directory"
        assert(len(don_input_path) > 0 or option == 0), "Run option requires Donor stack filename"

        # Find the epsilon values
        if (len(eps_entry_str.get()) > 0):
            eps = eps_entry_str.get()
        else:
            raise IOError("epsilon value must be provided")

        # Get config parameters from the gui
        frames = range_str.get()
        resolution = res_str.get()
        parallel = par_state.get()
        nwindow = nwindow_str.get()
        crop = crop_str.get()
        register = register_state.get()
        union = union_state.get()
        acceptor_bleach_range = acc_bleach_str.get()
        donor_bleach_range = don_bleach_str.get()
        fit = bleach_fit.get()

        # Create directory for config file and place empty config file in it
        new_path = dire+'/'+filename
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        cfname = new_path+'/Config_'+filename+'.cfg'
        copyfile('/'.join([dire,'Config_temp.cfg']),cfname)

        # Fill config file with gui parameters
        with open(cfname,'r') as f:
            filedata = f.read()

        filedata = filedata.replace("input_path =",f"input_path = {input_path}")\
            .replace("filename =",f"filename = {filename}")\
            .replace("extension =", f"extension = {extension}")\
            .replace("frames =", f"frames = {frames}")\
            .replace("resolution =", f"resolution = {resolution}")\
            .replace("parallel =", f"parallel = {parallel}")\
            .replace("nwindow =", f"nwindow = {nwindow}")\
            .replace("eps =", f"eps = {eps}")\
            .replace("crop =", f"crop = {crop}")\
            .replace("register =", f"register = {register}")\
            .replace("union =", f"union = {union}")\
            .replace("acceptor_bleach_range =", f"acceptor_bleach_range = {acceptor_bleach_range}") \
            .replace("donor_bleach_range =", f"donor_bleach_range = {donor_bleach_range}") \
            .replace("fit =", f"fit = {fit}")\
            .replace("option =", f"option = {option}")

        with open(cfname, 'w') as file:
            file.write(filedata)

        # Run pipeline using newly created config file
        parameter_extraction.main_extract(cfname,bool(tiff_state.get()),verbose,bool(h5_state.get()),bool(anim_state.get()))

    else:
        # Get existing config file from gui and check validity
        cfname_exist = con_str.get()
        assert(cfname_exist[-3:] == 'cfg'), "Please select a valid config file (.cfg extension)"

        # Replace run option in existing config file
        with open(cfname_exist,'r') as f:
            filedata = f.read()

        filedata = re.sub("option = \d", f"option = {option}",filedata)

        with open(cfname_exist, 'w') as file:
            file.write(filedata)

        # Run pipeline using existing config file
        parameter_extraction.main_extract(cfname_exist,bool(tiff_state.get()),verbose,bool(h5_state.get()),bool(anim_state.get()))

def main_gui():
    root = tk.Tk()
    root.title('fret-ibra')
    style = ttk.Style(root)
    style.theme_use('alt')

    dire = os.getcwd()

    frm_head0 = tk.Frame(root,padx=5,pady=1)
    frm_head0.pack(side="top",fill="x",expand=True)
    tk.Label(frm_head0,text='Upload Existing Config File', fg='slate grey',font=40).grid(sticky="W",row=5,column=1)

    frm0 = tk.Frame(root,padx=5,pady=1)
    frm0.pack(side="top",fill="x",expand=True)
    ttk.Label(frm0,text='Config Filename'.ljust(21), font=40).grid(sticky="W",row=5,column=1)
    con_str = tk.StringVar(root)
    config_entry = tk.Entry(frm0, textvariable = con_str, width=50)
    config_entry.grid(sticky="W",row=5,column=2)
    ttk.Button(frm0, text="Choose File", width=12, command=lambda:open_file(config_entry,dire)).grid(sticky="W",row=5,column=3,padx=2)

    frm_line0 = tk.Frame(root,padx=5,pady=1)
    frm_line0.pack(side="top", fill="x", expand=True)
    canvas = tk.Canvas(frm_line0, width=500, height=10)
    canvas.pack()
    canvas.create_line(0, 5, 600, 5, fill="black", tags="line")

    frm_head1 = tk.Frame(root,padx=5,pady=1)
    frm_head1.pack(side="top",fill="x",expand=True)
    tk.Label(frm_head1,text='File Input', fg='slate grey', font=40).grid(sticky="W",row=5,column=1)

    frm1 = tk.Frame(root,padx=5,pady=1)
    frm1.pack(side="top",fill="x",expand=True)
    ttk.Label(frm1,text='Acceptor Filename'.ljust(21), font=40).grid(sticky="W",row=5,column=1)
    acc_str = tk.StringVar(root)
    acceptor_entry = tk.Entry(frm1, textvariable = acc_str, width=50)
    acceptor_entry.grid(sticky="W",row=5,column=2)
    ttk.Button(frm1, text="Choose File", width=12, command=lambda:open_file(acceptor_entry,dire)).grid(sticky="W",row=5,column=3,padx=2)

    frm2 = tk.Frame(root,padx=5,pady=1)
    frm2.pack(side="top",fill="x",expand=True)
    ttk.Label(frm2,text='Donor Filename'.ljust(21), font=40).grid(sticky="W",row=5,column=1)
    don_str = tk.StringVar(root)
    donor_entry = ttk.Entry(frm2, textvariable = don_str, width=50)
    donor_entry.grid(sticky="W", row=5, column=2)
    ttk.Button(frm2, text="Choose File", width=12, command=lambda:open_file(donor_entry,dire)).grid(sticky="W",row=5,column=3,padx=2)

    frm_line1 = tk.Frame(root,padx=5,pady=1)
    frm_line1.pack(side="top", fill="x", expand=True)
    canvas = tk.Canvas(frm_line1, width=500, height=10)
    canvas.pack()
    canvas.create_line(0, 5, 600, 5, fill="black", tags="line")

    frm_head2 = tk.Frame(root,padx=5,pady=1)
    frm_head2.pack(side="top",fill="x",expand=True)
    tk.Label(frm_head2,text='Processing Parameters', fg='slate grey', font=40).grid(sticky="W",row=5,column=1)

    frm4 = tk.Frame(root,padx=5,pady=1)
    frm4.pack(side="top",fill="x",expand=True)
    ttk.Label(frm4, text='Frame Range'.ljust(21), font=40).grid(sticky="W",row=5,column=1)
    range_str = tk.StringVar(root)
    range_str.set('1:1')
    range_entry = ttk.Entry(frm4,textvariable = range_str, width=10)
    range_entry.grid(sticky="W",row=5,column=3,padx=2)

    frm5 = tk.Frame(root,padx=5,pady=1)
    frm5.pack(side="top",fill="x",expand=True)
    ttk.Label(frm5, text='Resolution'.ljust(21), font=40).grid(sticky="W",row=5,column=1)
    res_str = tk.StringVar(root)
    res_str.set(8)
    res_entry = ttk.Entry(frm5, textvariable = res_str, width=10)
    res_entry.grid(sticky="W",row=5,column=3,padx=2)

    frm6 = tk.Frame(root,padx=5,pady=1)
    frm6.pack(side="top",fill="x",expand=True)
    ttk.Label(frm6,text='Parallel'.ljust(21), font=40).grid(sticky="W",row=5,column=1)
    par_state = tk.IntVar(root)
    par_state.set(0)
    par_check = ttk.Checkbutton(frm6,variable=par_state)
    par_check.grid(sticky="W",row=5,column=2,padx=2)

    frm_line2 = tk.Frame(root,padx=5,pady=1)
    frm_line2.pack(side="top", fill="x", expand=True)
    canvas = tk.Canvas(frm_line2, width=500, height=10)
    canvas.pack()
    canvas.create_line(0, 5, 600, 5, fill="black", tags="line")

    frm_head3 = tk.Frame(root,padx=5,pady=1)
    frm_head3.pack(side="top",fill="x",expand=True)
    tk.Label(frm_head3,text='Background Subtraction Parameters', fg='slate grey', font=40).grid(sticky="W",row=5,column=1)

    frm7 = tk.Frame(root,padx=5,pady=1)
    frm7.pack(side="top",fill="x",expand=True)
    ttk.Label(frm7,text='Number of Windows'.ljust(21), font=40).grid(sticky="W",row=5,column=1)
    nwindow_str = tk.StringVar(root)
    nwindow_str.set(40)
    nwindow_entry = ttk.Entry(frm7, textvariable = nwindow_str, width=10)
    nwindow_entry.grid(sticky="W",row=5,column=2,padx=2)

    frm8 = tk.Frame(root,padx=5,pady=1)
    frm8.pack(side="top",fill="x",expand=True)
    ttk.Label(frm8,text='Epsilon'.ljust(21), font=40).grid(sticky="W",row=5,column=1)
    eps_entry_str = tk.StringVar(root)
    eps_entry_str.set(0.01)
    epsilon_entry = ttk.Entry(frm8, textvariable=eps_entry_str, width=10)
    epsilon_entry.grid(sticky="W",row=5,column=2,padx=2)

    #eps_check_str = tk.StringVar(root)
    #epsilon_check = ttk.Checkbutton(frm8, variable=eps_check_str, text='Use calculated value')
    #epsilon_check.grid(sticky="W",row=5,column=3)

    frm_line3 = tk.Frame(root,padx=5,pady=1)
    frm_line3.pack(side="top", fill="x", expand=True)
    canvas = tk.Canvas(frm_line3, width=500, height=10)
    canvas.pack()
    canvas.create_line(0, 5, 600, 5, fill="black", tags="line")

    frm_head4 = tk.Frame(root,padx=5,pady=1)
    frm_head4.pack(side="top",fill="x",expand=True)
    tk.Label(frm_head4,text='Ratio Processing Parameters', fg='slate grey', font=40).grid(sticky="W",row=5,column=1)

    frm9 = tk.Frame(root,padx=5,pady=1)
    frm9.pack(side="top",fill="x",expand=True)
    ttk.Label(frm9,text='Crop'.ljust(21), font=40).grid(sticky="W",row=5,column=1)
    crop_str = tk.StringVar(root)
    crop_str.set('0,0,0,0')
    crop_entry = ttk.Entry(frm9, textvariable = crop_str, width=10)
    crop_entry.grid(sticky="W",row=5,column=2)

    frm10 = tk.Frame(root,padx=5,pady=1)
    frm10.pack(side="top",fill="x",expand=True)
    ttk.Label(frm10,text='Register'.ljust(21), font=40).grid(sticky="W",row=5,column=1)
    register_state = tk.IntVar(root)
    register_state.set(1)
    register_check = ttk.Checkbutton(frm10,variable=register_state)
    register_check.grid(sticky="W",row=5,column=2)

    frm11 = tk.Frame(root,padx=5,pady=1)
    frm11.pack(side="top",fill="x",expand=True)
    ttk.Label(frm11,text='Union'.ljust(21), font=40).grid(sticky="W",row=5,column=1)
    union_state = tk.IntVar(root)
    union_state.set(1)
    union_check = ttk.Checkbutton(frm11,variable=union_state)
    union_check.grid(sticky="W",row=5,column=3)

    frm_line4 = tk.Frame(root,padx=5,pady=1)
    frm_line4.pack(side="top", fill="x", expand=True)
    canvas = tk.Canvas(frm_line4, width=500, height=10)
    canvas.pack()
    canvas.create_line(0, 5, 600, 5, fill="black", tags="line")

    frm_head5 = tk.Frame(root,padx=5,pady=1)
    frm_head5.pack(side="top",fill="x",expand=True)
    tk.Label(frm_head5,text='Bleach Processing Parameters', fg='slate grey', font=40).grid(sticky="W",row=5,column=1)

    frm12 = tk.Frame(root,padx=5,pady=1)
    frm12.pack(side="top",fill="x",expand=True)
    ttk.Label(frm12,text='Acceptor Bleach Range'.ljust(21), font=40).grid(sticky="W",row=5,column=1)
    acc_bleach_str = tk.StringVar(root)
    acc_bleach_str.set('1:1')
    acc_bleach_file_entry = ttk.Entry(frm12, textvariable = acc_bleach_str, width=10)
    acc_bleach_file_entry.grid(sticky="W",row=5,column=2)

    frm13 = tk.Frame(root,padx=5,pady=1)
    frm13.pack(side="top",fill="x",expand=True)
    ttk.Label(frm13,text='Donor Bleach Range'.ljust(21), font=40).grid(sticky="W",row=5,column=1)
    don_bleach_str = tk.StringVar(root)
    don_bleach_str.set('1:1')
    don_bleach_file_entry = ttk.Entry(frm13, textvariable = don_bleach_str, width=10)
    don_bleach_file_entry.grid(sticky="W",row=5,column=2)

    frm14 = tk.Frame(root,padx=5,pady=1)
    frm14.pack(side="top",fill="x",expand=True)
    ttk.Label(frm14,text='Bleach Fit Type'.ljust(21), font=40).grid(sticky="W",row=5,column=1)
    bleach_fit = tk.StringVar(root)
    bleach_fit.set('linear')
    bleach_radio_linear = tk.Radiobutton(frm14, text="linear", variable=bleach_fit, value='linear')
    bleach_radio_linear.grid(sticky="W",row=5,column=2)
    bleach_radio_quad = tk.Radiobutton(frm14, text="quadratic", variable=bleach_fit, value='quadratic')
    bleach_radio_quad.grid(sticky="W",row=5,column=3)

    frm_line5 = tk.Frame(root,padx=5,pady=1)
    frm_line5.pack(side="top", fill="x", expand=True)
    canvas = tk.Canvas(frm_line5, width=500, height=10)
    canvas.pack()
    canvas.create_line(0, 5, 600, 5, fill="black", tags="line")

    frm_head6 = tk.Frame(root,padx=5,pady=1)
    frm_head6.pack(side="top",fill="x",expand=True)
    tk.Label(frm_head6,text='Run Parameters', fg='slate grey', font=40).grid(sticky="W",row=5,column=1)

    frm15 = tk.Frame(root,padx=5,pady=1)
    frm15.pack(side="top",fill="x",expand=True)
    ttk.Label(frm15,text='Output Options'.ljust(21), font=40).grid(sticky="W",row=5,column=1)
    tiff_state = tk.IntVar(root)
    tiff_state.set(1)
    tiff_output_check = ttk.Checkbutton(frm15,text='tiff output', variable=tiff_state)
    tiff_output_check.grid(sticky="W",row=5,column=2)

    h5_state = tk.IntVar(root)
    h5_state.set(1)
    h5_output_check = ttk.Checkbutton(frm15,text='h5 output',variable=h5_state)
    h5_output_check.grid(sticky="W",row=5,column=3)

    anim_state = tk.IntVar(root)
    anim_state.set(1)
    anim_output_check = ttk.Checkbutton(frm15,text='animation output',variable=anim_state)
    anim_output_check.grid(sticky="W",row=5,column=4)

    frm16 = tk.Frame(root,padx=5,pady=1)
    frm16.pack(side="top",fill="x",expand=True)
    ttk.Label(frm16,text='Run Options'.ljust(21), font=40).grid(sticky="W",row=5,column=1)
    option_line = tk.StringVar(root)
    option_dict = {
        'Background subtraction (acceptor) only' : '0',
        'Background subtraction (donor) only' : '1',
        'Ratio processing' : '2',
        'Background subtraction (both channels) + Ratio Processing' : '3',
        'Bleach correction' : '4'
    }
    option_choices = list(option_dict.keys())
    option_state = ttk.OptionMenu(frm16, option_line, option_choices[0], *option_choices)
    option_state.grid(sticky="W",row=5,column=2)

    frm17 = tk.Frame(root,padx=5,pady=1)
    frm17.pack(side="top",fill="x",expand=True)
    ttk.Button(frm17, text="Run", command=lambda:get_variables(dire,con_str,acc_str,don_str,range_str,res_str,par_state,nwindow_str\
                                                                  ,eps_entry_str,crop_str,register_state,union_state,acc_bleach_str\
                                                                  ,don_bleach_str,bleach_fit,tiff_state,h5_state,anim_state,option_dict,option_line), width=10).grid(row=5,column=1,padx=270)

    frm_head7 = tk.Frame(root,padx=5,pady=1)
    frm_head7.pack(side="top",fill="x",expand=True)
    tk.Label(frm_head7,text='For detailed parameter descriptions visit: github.com/gmunglani/fret-ibra/blob/gui/examples/Tutorial.md' ,font=40).grid(sticky="W",row=5,column=1)

    root.mainloop()
